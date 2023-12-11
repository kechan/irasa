from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union

from pathlib import Path
import json, re

@dataclass(frozen=True)
class SimplifiedMessage:
    '''
    role should be extracted from author field e.g. {'role': 'user', 'name': None, 'metadata': {}}
    content should be extracted from content.parts field e.g. {'content_type': 'text', 'parts': ['blah blah']},
    '''
    role: str      # can be system, user, tool, or assistant #TODO: try use Enum
    content: str
    id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass(frozen=True)
class Message:
    id: str
    content: Dict  # this is {'content_type': 'text', 'parts': ['blah blah']},

    status: str
    end_turn: bool
    weight: float
    author: Optional[Dict] = None  # this is {'role': 'user', 'name': None, 'metadata': {}}
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    recipient: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def simplify(self) -> SimplifiedMessage:
        """
        Converts the current Message instance to its SimplifiedMessage representation.
        :return: The SimplifiedMessage representation of the current Message instance.
        """

        # Extracting role from the author field, whether it's a string or dictionary
        role = self.author if isinstance(self.author, str) else self.author.get("role")
        
        # Extracting the text value directly from the parts key within the content dictionary
        text_content = self.content["parts"][0] if "parts" in self.content else ""
        
        return SimplifiedMessage(role=role, content=text_content, id=self.id)

@dataclass(frozen=True)
class Thread:
    """
    Represents a linear sequence of messages in a conversation.
    Can be used to capture both simplified and regular message sequences.
    """
    messages: Union[List[Message], List[SimplifiedMessage]]  # A thread is either list of messages or simplified messages

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index: Union[int, slice]) -> Union[Message, SimplifiedMessage]:
        return self.messages[index]
    
    def __iter__(self):
        return iter(self.messages)

    def to_dict(self):
        return {"messages": [message.to_dict() for message in self.messages]}

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)


@dataclass(frozen=True)
class HistoricalThreads:
    """
    Due to the branching structure of a conversation, this is the List representation of that tree 
    (kind of depth first search if you search oldest branch first, it will map exactly to the real sequence
    of conversation the user made with chatGPT)
    
    It captures the concept of a conversation being a collection of threads, 
    where each thread is a sequence of messages.

    TODO: One thing to ensure, which seems implicit in your setup, is that the conversation.completions dictionary 
    is already ordered based on the sequence of interactions. If it isn't, this method might not return threads in 
    the strictly chronological order you intend. If that order is guaranteed elsewhere or is a property of the data source, 
    then everything looks good.
    """
    threads: List[Thread]

    def __len__(self):
        return len(self.threads)

    def __getitem__(self, index: int) -> Thread:
        return self.threads[index]
    
    def __iter__(self):
        return iter(self.threads)
    
@dataclass(frozen=True)
class Completion:
    id: str
    message: Message
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Conversation:
    completions: Dict[str, Completion]
    title: Optional[str] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    moderation_results: Optional[Dict[str, Any]] = None
    current_node: Optional[str] = None
    plugin_ids: Optional[List[str]] = None
    id: Optional[str] = None

    def flattened_simplified_messages(self, as_dict=False) -> Union[List[SimplifiedMessage], List[Dict]]:
        """
        Returns a flattened list of SimplifiedMessage objects from the conversation.
        This disregards the parent-child relation between messages.
        """

        messages = [completion.message.simplify() for completion in self.completions.values()]
        if as_dict:
            return [message.__dict__ for message in messages]
        return messages
    
    def flattened_messages(self, as_dict=False) -> Union[List[Message], List[Dict]]:
        """
        Returns a flattened list of Message objects from the conversation.
        This disregards the parent-child relation between messages.
        """

        messages = [completion.message for completion in self.completions.values()]
        if as_dict:
            return [message.__dict__ for message in messages]
        return messages

    def get_historical_threads(self, as_dict=False) -> Union[HistoricalThreads, List[List[Dict]]]:
        """
        Constructs and returns a HistoricalThreads object for a given conversation.

        The method processes the conversation and breaks it into separate threads. Each thread is represented as a list 
        of SimplifiedMessage objects in the order they appear in the conversation. If any message in the conversation 
        has multiple children (indicative of a "Regenerate" action taken by a user), this leads to branching. These branches 
        are treated as separate threads.

        The threads in the resulting HistoricalThreads object are ordered chronologically, with earlier threads indicating 
        older interactions and the last thread representing the most recent interactions in the conversation.
        """

        # 0) sanity check chronlogical assumtion 
        if not self.is_chronological:
            # raise ValueError("The conversation is not chronological. Please sort the completions by message.create_time.")
            print("Warning: The conversation is not chronological. Please sort the completions by message.create_time, or check results")

        # 1) construct a Dict whose keys are the completion id (the parent) and
        # values completion (or message) ids of its children (i.e. List of Ids)
        parent_child_dict = {completion.id: completion.children for _, completion in self.completions.items()}

        # 2) Check how many "root" node are there
        all_nodes = set(self.completions.keys())
        child_nodes = {child for _, children in parent_child_dict.items() for child in children}
        potential_roots = all_nodes - child_nodes  # node that never appear as a child anywhere

        if len(potential_roots) != 1:
            print(f"Warning: There are {len(potential_roots)} potential root nodes. Please double check all results.")

        # 3) Use recursion to walk the tree and construct the threads in terms List of List of Ids 
        # Strong assumptions: the list of completions are arranged in chronological order
        # and the first completion is the root of the tree
        def build_history(m_id) -> List[List[str]]:
          '''
          Given a message id, return all the threads following from this including itself.
          '''
          if parent_child_dict[m_id] == []:   # no child, one thread and one message with for this completion
              return [[m_id]]
          else:
              all_history = []
              for child_id in parent_child_dict[m_id]:
                  history = build_history(child_id)
                  for h in history:
                      h.insert(0, m_id)
                  all_history.extend(history)
              return all_history
          
        root_m_id = list(self.completions.keys())[0]     # TODO: is it possible to make this more robust?
        # all_threads = build_history(root_m_id)
        all_threads = []
        for root_m_id in potential_roots:
            all_threads.extend(build_history(root_m_id))

        # construct threads from the list of ids
        threads = []
        for thread in all_threads:
            messages=[]
            for m_id in thread:
                messages.append(self.completions[m_id].message.simplify())
            threads.append(Thread(messages=messages))

        historical_threads = HistoricalThreads(threads=threads)

        if as_dict:
            return [[message.__dict__ for message in thread] for thread in historical_threads]
        
        return historical_threads

    @property
    def is_chronological(self) -> bool:
        """
        Check if the completions in the given conversation are in chronological order based on the create_time field.

        Parameters:
        - conversation: A Conversation instance.

        Returns:
        - True if the completions are in chronological order, otherwise False.
        """
        
        # Extract create_time for each completion message
        #   create_times = [completion.message.create_time for _, completion in self.completions.items()]
        create_times = [completion.message.create_time for _, completion in self.completions.items() if completion.message.create_time is not None]

        # Check if the list of create_time values is sorted
        return all(t1 <= t2 for t1, t2 in zip(create_times, create_times[1:]))

    
class ChatGPTConversations:
    """
    This class encapsulates conversations with ChatGPT. Notably, a conversation can comprise multiple threads, 
    giving the messages a tree-like structure. This branching arises when a user clicks the "Regenerate" button for a past 
    point in the conversation.

    params: json_path: Path to the JSON dump of conversations from ChatGPT.
    """
    def __init__(self, json_path: Union[str, Path], use_regex_indexing=True):
        with open(json_path, 'r') as f:
          data = json.load(f)
        self._conversations = [self.parse_conversation(conv_data) for conv_data in data]
        self._titles = None    # lazy load this

        self.use_regex_indexing = use_regex_indexing

    def __getitem__(self, key: Union[int, slice, str]) -> Union[Conversation, List[Conversation]]:
        """
        Returns a conversation by index, slice, or title.
        """
        if isinstance(key, int) or isinstance(key, slice):
            return self._conversations[key]
        
        # for str indexing, use regex pattern match on the title
        if self.use_regex_indexing:
            matched_conversations = [conversation for conversation in self._conversations if re.search(key, conversation.title or '', re.IGNORECASE)]
        else:
            # for str indexing, use case-insensitive exact match on the title
            matched_conversations = [conversation for conversation in self._conversations if conversation.title and key.lower() == conversation.title.lower()]

        if matched_conversations:        
            return matched_conversations
        
        raise KeyError(f"No conversation found with title '{key}'")
    

    def __iter__(self):
        return iter(self._conversations)

    def __len__(self):
        return len(self._conversations)
    

    def __call__(self):
        return self._conversations
    
    @property
    def titles(self) -> List[str]:
        if self._titles is None:
            self._titles = [conversation.title for conversation in self._conversations]
        return self._titles
    
    # various basic parsing methods

    def parse_simplified_message(self, data: Dict[str, Any]) -> SimplifiedMessage:
        return SimplifiedMessage(
            role=data["role"],
            content=data["content"]
        )

    def parse_message(self, data: Dict[str, Any]) -> Message:
        return Message(
            id=data["id"],
            author=data.get("author"),
            create_time=data.get("create_time"),
            update_time=data.get("update_time"),
            content=data["content"],
            status=data["status"],
            end_turn=data["end_turn"],
            weight=data["weight"],
            metadata=data.get("metadata"),
            recipient=data["recipient"]
        )

    def parse_completion(self, completion_id: str, data: Dict[str, Any]) -> Optional[Completion]:
        if not data.get("message"):
            return None
        
        return Completion(
            id=completion_id,
            message=self.parse_message(data["message"]),
            parent=data.get("parent"),
            children=data.get("children", [])
        )

    def parse_conversation(self, data: Dict[str, Any]) -> Conversation:
        completions_data = {
            comp_id: self.parse_completion(comp_id, comp_data)
            for comp_id, comp_data in data["mapping"].items()
        }
        completions_data = {k: v for k, v in completions_data.items() if v}
        
        return Conversation(
            completions=completions_data,
            title=data.get("title"),
            create_time=data.get("create_time"),
            update_time=data.get("update_time"),
            moderation_results=data.get("moderation_results"),
            current_node=data.get("current_node"),
            plugin_ids=data.get("plugin_ids"),
            id=data.get("id")
        )





