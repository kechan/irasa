import tiktoken

def get_num_tokens(string: str, encoding_name='cl100k_base') -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  string = string.replace('<|endofprompt|>', '<|end0fprompt|>')  # Replace the special token
  string = string.replace('<|endoftext|>', '<|end0ftext|>')  # Replace the special token
  string = string.replace('<|fim_prefix|>', '<|f1m_prefix|>')  # Replace the special token
  string = string.replace('<|fim_middle|>', '<|f1m_middle|>')  # Replace the special token
  string = string.replace('<|fim_suffix|>', '<|f1m_suffix|>')  # Replace the special token
  num_tokens = len(encoding.encode(string))
  return num_tokens


def truncate(string: str, encoding_name='cl100k_base', n_token=50):
  """ Truncates a string to n_token """
 
  encoding = tiktoken.get_encoding(encoding_name)
  tokens = encoding.encode(string)
  truncated_tokens = tokens[:n_token]
  truncated_string = encoding.decode(truncated_tokens)
  return truncated_string

def truncate_head_tail(string: str, encoding_name='cl100k_base', n_token=50) -> str:
  """ Truncates a string to n_token by keeping tokens from the head and tail with '...' in between. """
  
  if n_token <= 3:  # Ensure a reasonable limit to maintain head and tail tokens
    raise ValueError("n_token should be greater than 3 to maintain head and tail structure.")
  
  encoding = tiktoken.get_encoding(encoding_name)
  tokens = encoding.encode(string)
  
  if len(tokens) <= n_token:  # If the string is already shorter than n_token, return as is
      return string

  # The -3 accounts for the tokens from "..."
  half_length = (n_token - 3) // 2
  
  head_tokens = tokens[:half_length]
  tail_tokens = tokens[-half_length:]
  
  truncated_string = encoding.decode(head_tokens) + "..." + encoding.decode(tail_tokens)
  
  return truncated_string