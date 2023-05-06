import os
# import OpenAI
OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
prompt = """
As a developer I have to built a system in which I will feed the input document and want to questions on that data. Can you please recommend me what services I have to use?
"""

print(llm(prompt=prompt))