import openparse

basic_doc_path = "data/parallel_algo.pdf"
parser = openparse.DocumentParser()
parsed_basic_doc = parser.parse(basic_doc_path)

for node in parsed_basic_doc.nodes:
    print(node)