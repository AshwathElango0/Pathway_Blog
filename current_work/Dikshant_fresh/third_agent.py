from agent import Agent
from pydantic import BaseModel
from typing import List, Union, Literal
class RetrieverAgent:
    
    def __init__(self, retriever: Retriever, **kwargs):
        self.retriever = retriever
        self.agent = Agent(**kwargs)
        self._retrieved_data=None
        self._retrieved_metadata=None
        self.agent.add_tools([retrieve])
        self._k = 5
        self._query = None

        
    def retrieve(query : str):
        """This Function is used if data retrieval is required. Either because the existing data doesn't provide sufficient information or the user wants to explore more."""
        return self.retriever.retrieve(query, k = _k)
    

    
    def retrieve_docs(self, user_input):
        i = 0
        turns = 5
        while i < turns:
            res = self.agent.send_request(user_input)
            if res.suggests_tool_call:
                self._retrieved_data, self._retrieved_metadata = self.agent.execute_tool(res)
            else:
                break
            self._k *= 2
            i+= 1
            
        self._k = 5
        self._query = None
        retrieved, metadata = self._retrieved_data, self._retrieved_metadata
        self._retrieved_data = None
        self._retrieved_metadata = None
        return query, retrieved, metadata
    