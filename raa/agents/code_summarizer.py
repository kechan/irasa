from ..raa import RAA
import pyperclip
import tarfile, io, json, re
import pandas as pd
from pathlib import Path
from ..utils.tokens import get_num_tokens
from ..models.chatgpt_conversations import ChatGPTConversations
from realestate_core.common.utils import join_df

# handle tarfile utilities to tar up a git repo
def create_tarfile_of_git_repo(repo_path: Path, tarfile_path: Path):
  with tarfile.open(tarfile_path, "w:gz") as tar:
    tar.add(repo_path, arcname=repo_path.name, filter=lambda x: None if '.git' in x.name else x)


def filter_tar_gz(input_path: str, output_path: str, keep_all=False, keep_extensions=['.py', '.ipynb']):
    # In-memory file for the new tarball
    new_tar_data = io.BytesIO()

    # Open the existing tar.gz file
    with tarfile.open(input_path, 'r:gz') as tar:
        # Open a new tar.gz for writing
        with tarfile.open(fileobj=new_tar_data, mode='w:gz') as new_tar:
            # Iterate through each member in the original tarball
            for member in tar.getmembers():
                # print(Path(member.name).name)
                if ".git/" in member.name or Path(member.name).name.startswith('.git'):  # always skip .git*
                    print(f"skipping {member.name}")
                    continue            

                if not keep_all:
                    # If the member's name ends with keep_extensions, add it to the new tarball
                    if any([member.name.endswith(ext) for ext in keep_extensions]):
                        new_tar.addfile(member, fileobj=tar.extractfile(member))
                else:
                    new_tar.addfile(member, fileobj=tar.extractfile(member))
                        
                        

    # Write the in-memory tarball to the output file
    with open(output_path, 'wb') as f:
        f.write(new_tar_data.getvalue())


class LongCodeSummarizingAgent:
  """ For Code Interpreter """
  
  def __init__(self, system_prompt=None):
    if system_prompt is None:
      self.system_prompt = """You are a python expert. I have some python code I would like your help with summarizing.
But first, lets set up your env. You will need a custom version of tiktoken and a class
to help with parsing code into sensible blocks.

Without doing anything else, please first run PythonCodeBlockChunkingStrategy.py
then do:

from tiktoken.core import Encoding
import pickle
# Pickle load cl100k_base_params into params before moving onto next lines of code
enc = Encoding(**params, use_pure_python=True)

verify by running:

enc.encode("!"), enc.encode("hello world")

and then run:

chunk_strategy = PythonCodeBlockChunkingStrategy(chunk_token_size=1024)
chunk_strategy.chunk_text("def foo():\n  print('hello world')")
"""
    else:
      self.system_prompt = system_prompt

    self.raa = RAA(sys_prompt=self.system_prompt)    # chatgpt completion API wrapped up as an agent

  def initial_setup(self):
    pyperclip.copy(self.system_prompt)
    print("initial setup prompt copied to clipboard")

  def work_on(self, tar_filename):
    """
    summarize all .py in the uploaded tar
    """
    user_prompt=f"""
**The goal is to methodically analyze and manually summarize the content of Python files from an 
uploaded tar.gz, ensuring clarity and conciseness while capturing essential information for the purpose of vector embedding.**:

1. File Initialization:
   - Untar the the uploaded {tar_filename} 
   - If `*python_file_list.txt` is uploaded, use it. It will contain rows in the format `file<N>, filepath` where filepath is the relative path respect to the tar directory.
   - Analyze the directory structure and enumerate all `.py` files (skipping `._*`, `.__*` and `__init__.py` and anything under `.git`). Save this list to `*python_file_list.txt` in the format `file0, /path/to/main.py`.
   - file<N> will be used as reference to the actual filepath

2. Sanity Checks:
   - Confirm the filename of `file0` and match with `*python_file_list.txt`.
   - Ensure paths for all files (`file0` to `fileN`) exist.
   - let me download *python_file_list.txt

3. File Analysis:
   - For each `.py` file:
      1. Use "code_chunks = chunk_strategy.chunk_text(content)" where content is the long string content of the file, to get a list of code_chunks.
      2. Then manually using your inherent understanding without resorting to parsing tools or heuristic methods. Focus on:
         * Class/method signatures
         * Comments
         * Code structure
      3. Without echoing back the code in your response (this part is slow), describe in plain ascii your comprehension of that block of code. 
      4. save/append the summary to a file named `file<N>_summary.txt` (such that it contains everything in 3. for that file)
      5. After all code blocks are examined:
         1. manually draft a summary relying solely on your understanding and analysis without using any automated or heuristic methods. Limit the summary to 512 words.
      
         ** Use the following guidelines for summary:
            * Remove Introductory Phrases: Avoid phrases like "This file...", "In this module...", etc.
            * Remove Transitional Words/Phrases: Exclude "Furthermore", "Lastly", etc.
            * Remove Markdown and Formatting: Strip markdown symbols and formatting characters.
            * Streamline Verbiage: Use active voice and concise verbs.
            * Final Cleanup: Ensure clarity, removing stray punctuation or symbols, retaining the essence.

         2.Cross-check the filename with `*python_file_list.txt`.
             - [FORMAT] Important: Output your final result in this format: 
                [filename]filepath</filename><summary>Summary text.[/summary]

         3. Confirm again filename and summary format are also consistent with the pattern shown in `code_summary_anchor_template.txt`.
         4. [ECHO] the [filename]whatever[/filename] and [summary]whatever[/summary] stuff back to me.
         5. if everything is confirmed, move to next file.


- Use the following prompts:
  `>`: Move onto the next code block of the same file.
  `>>`: Move to the next file.
  `>>>`: Skip the current file and move to the subsequent one.
  `aa`: save/append intermediate summary for current code block to file<N>_summary.txt if file<N> is the current file.
  `s`: Summarize the current file immediately without further segments.
  `c`: Sanity check if filename is consistent with whatever in `*python_file_list.txt` and filename/summary format is consistent with specified in code_summary_anchor_template.txt (read it first).
  `d <filename>`: To download the latest version of <filename> 
"""
    pyperclip.copy(user_prompt)
    print("main instruction prompt copied to clipboard")

  def env_reset_prompt(self):
    """
    prompt for env reset
    """
    user_prompt = """
run reset.py
and then
%reset
if you encounter any erors, just remove the stuff added to sys path to your best ability.
then make sure /mnt/data is empty. 

Finally, forget all instructions I have given so far, we will start from scratch.
"""
    pyperclip.copy(user_prompt)
    print("env reset prompt copied to clipboard")


  def extract_summaries_from_chatgpt_json(self, json_file, conversation_title_pat=None, python_file_list: str = None) -> pd.DataFrame:
    """
    extract summaries from chatgpt json file

    :param json_file: path to chatgpt conversation json file
    :param conversation_title_pat: regex pattern to filter conversations
    :param python_file_list: path to python_file_list.txt, which is sort of manifest for long code filepaths
    """
    conversations = ChatGPTConversations(json_path = json_file)
    if conversation_title_pat is not None:
      conversations = conversations[conversation_title_pat]

    # short conversations by create_time (will keop='last' for any dedup op)
    conversations = sorted(conversations, key=lambda x: x.create_time)

    all_messages = []
    for c in conversations:
      threads = c.get_historical_threads(as_dict=False)
      all_messages.extend(threads[-1].to_dict()['messages'])  # only latest thread

    conversations_df = pd.DataFrame(all_messages)  # role, content, and id

    # Sequentially pair up <filename/> and <summary/> tags
    paired_filename_summaries = []
    for content in conversations_df.content:
        filenames_in_content = re.findall(r'[filename](.*?)[/filename]', content, re.DOTALL)
        summaries_in_content = re.findall(r'[summary](.*?)[/summary]', content, re.DOTALL)
        
        # Ensure that we have matching number of filename and summary tags for each content entry
        if len(filenames_in_content) == len(summaries_in_content):
            for i in range(len(filenames_in_content)):
                paired_filename_summaries.append((filenames_in_content[i].strip(), summaries_in_content[i].strip()))
        else:
            print(f'Inconsistent number of filename and summary tags for content: {content}')
            # raise ValueError("Number of filename and summary tags do not match for content")

    # Convert paired data to DataFrame
    summaries_df = pd.DataFrame(paired_filename_summaries, columns=['filepath', 'summary'])
    # drop rows that has summary of "Your plain text summary here."
    summaries_df.drop(index=summaries_df.q("summary == 'Your plain text summary here.'").index, inplace=True)

    summaries_df['n_tokens'] = summaries_df.summary.apply(get_num_tokens)
    summaries_df['filename'] = summaries_df.filepath.apply(lambda x: x.split('/')[-1])

    # remove root dir from filepath
    summaries_df['filepath'] = summaries_df.filepath.apply(lambda x: '/'.join(x.split('/')[1:]))

    summaries_df.drop_duplicates(subset=['filepath'], keep='last', inplace=True)

    # drop if filename doesnt end in .py
    # TODO: may expand to incl. .ipynb
    summaries_df.drop(index=summaries_df.q("filename.str.endswith('.py') == False").index, inplace=True)

    summaries_df.reset_index(drop=True, inplace=True)   # defrag index

    summaries_df = summaries_df[['filename', 'filepath', 'summary', 'n_tokens']]

    # some summary dont matchup with anything real, dont want them
    # consult openai_cookbook_python_file_list.txt

    file_list_df = pd.read_csv(python_file_list, header=None, names=['filename', 'filepath'])
    file_list_df['filepath'] = file_list_df.filepath.apply(lambda x: '/'.join(x.split('/')[1:]))   # remove root dir from filepath

    summaries_df = join_df(file_list_df[['filepath']], summaries_df[['filename', 'filepath', 'summary', 'n_tokens']], left_on='filepath', how='left')

    return summaries_df






class CodeSummarizingAgent:
  """ For Code Interpreter """
  
  def __init__(self, system_prompt=None):
    if system_prompt is None:
      self.system_prompt = """You are a python expert. I have some python code below delimited by
<code> and </code> that I would like your help with summarizing:
"""
    self.raa = RAA(sys_prompt=self.system_prompt)

  def work_on(self, filepath: str, code: str, max_new_tokens=512):
    # compute how many tokens in the code
    num_tokens = get_num_tokens(code)
    if num_tokens > 1024:
      raise ValueError(f"Code is too long. It has {num_tokens} tokens.")

    self.user_prompt = f"""The filepath is {filepath}\n<code>\n{code}\n</code>\nPlease keep the summary under {max_new_tokens} tokens.
[STYLE] The summary should be without using any markdown, special formatting, or symbols like backticks. Just use plain text. 

Use the following guidelines for summary:
    * Remove Introductory Phrases: Avoid phrases like "This file...", "In this module...", etc.
    * Remove Transitional Words/Phrases: Exclude "Furthermore", "Lastly", etc.
    * Remove Markdown and Formatting: Strip markdown symbols and formatting characters.
    * Streamline Verbiage: Use active voice and concise verbs.
    * Do not use bullet points or numbered lists.
    * Final Cleanup: Ensure clarity, removing stray punctuation or symbols, retaining the essence.

Please use the following [FORMAT]:
[filename]filepath[/filename][summary]Your summary text.[/summary].
"""
    self.final_prompt = self.system_prompt + self.user_prompt
    pyperclip.copy(self.final_prompt)
    print("Prompt copied to clipboard")

  def extract_summaries_from_chatgpt_json(self, json_file, conversation_title_pat=None) -> pd.DataFrame:
    """
    extract summaries from chatgpt json file
    """
    conversations = ChatGPTConversations(json_path = json_file)
    if conversation_title_pat is not None:
      conversations = conversations[conversation_title_pat]

    # short conversations by create_time (will keop='last' for any dedup op)
    conversations = sorted(conversations, key=lambda x: x.create_time)

    all_thread_messages = []
    for c in conversations:
      threads = c.get_historical_threads(as_dict=False)
      for thread in threads:
        all_thread_messages.extend(thread.to_dict()['messages'])

    conversations_df = pd.DataFrame(all_thread_messages)  # role, content, and id

    # Sequentially pair up <filename/> and <summary/> tags
    paired_filename_summaries = []
    for content in conversations_df.content:
        filenames_in_content = re.findall(r'[filename](.*?)[/filename]', content, re.DOTALL)
        summaries_in_content = re.findall(r'[summary](.*?)[/summary]', content, re.DOTALL)
        
        # Ensure that we have matching number of filename and summary tags for each content entry
        if len(filenames_in_content) == len(summaries_in_content):
            for i in range(len(filenames_in_content)):
                paired_filename_summaries.append((filenames_in_content[i].strip(), summaries_in_content[i].strip()))
        else:
            raise ValueError("Number of filename and summary tags do not match for content: {}".format(content))

    # Convert paired data to DataFrame
    summaries_df = pd.DataFrame(paired_filename_summaries, columns=['filepath', 'summary'])
    # drop rows that has summary of "Your plain text summary here."
    summaries_df.drop(index=summaries_df.q("summary == 'Your plain text summary here.'").index, inplace=True)

    summaries_df['n_tokens'] = summaries_df.summary.apply(get_num_tokens)
    summaries_df['filename'] = summaries_df.filepath.apply(lambda x: x.split('/')[-1])

    # remove root dir from filepath
    summaries_df['filepath'] = summaries_df.filepath.apply(lambda x: '/'.join(x.split('/')[1:]))

    summaries_df.drop_duplicates(subset=['filepath'], keep='last', inplace=True)

    # drop if filename doesnt end in .py
    # TODO: may expand to incl. .ipynb
    summaries_df.drop(index=summaries_df.q("filename.str.endswith('.py') == False").index, inplace=True)

    summaries_df.reset_index(drop=True, inplace=True)   # defrag index

    summaries_df = summaries_df[['filename', 'filepath', 'summary', 'n_tokens']]

    return summaries_df





