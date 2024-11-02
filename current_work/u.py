from litellm.utils import function_to_dict

def hello_world(x: int, u:int  =1, z: str="sf") -> int:
    """fsd"""
    return x + 1

print(function_to_dict(hello_world))