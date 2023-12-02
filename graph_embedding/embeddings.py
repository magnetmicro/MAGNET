import javalang
import os
import csv
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import nltk
import re



### Extract Methods source code using AST
def extract_method_bodies(java_file_path):
    with open(java_file_path, 'r') as file:
        file_content = file.read()

    # Parse the Java file
    tree = javalang.parse.parse(file_content)

    method_bodies = []

    for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
        for method_node in class_node.methods:            
            if method_node.body:
                start_line = method_node.position.line
                end_line = method_node.body[-1].position.line

                # Extract the lines for this method's body
                method_lines = file_content.splitlines()[start_line-1:end_line]

                # Reconstruct the body as a string
                method_body = '\n'.join(method_lines)

                # Store the method name with its body
                method_bodies.append((method_node.name, method_body))
    return method_bodies

### Create the feature matrix
def createEmbed(body):
        n=0
        allembadding=[]
        code_tokenss = tokenizer.tokenize(body)
        while(len(code_tokenss)>(n)*510):
            code_tokens=code_tokenss[(n)*510+1:(n+1)*510]
            tokens = [tokenizer.cls_token] +  code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
            embaddings= context_embeddings[0][0].tolist()
            allembadding.append(embaddings)
            n=n+1
        npmatrix=np.matrix(allembadding)
        meanmatrix=npmatrix.mean(0)
        arr=np.array(meanmatrix[0]).flatten()
        return arr

def createConstructor(name):
    constructor = f"public {name}() {{\n    super();\n}}"
    return constructor

def get_full_method_name(declaration, package_name, class_name):
    if package_name:
        full_name = f"{package_name}.{class_name}${declaration.name}"
    else:
        full_name = f"{class_name}${declaration.name}"
    return full_name

def extract_method_body(java_code, start_line, start_column):
    lines = java_code.splitlines()
    brace_count = 0
    method_body = ""
    found = False

    for i in range(start_line - 1, len(lines)):
        line = lines[i]
        if i == start_line - 1:
            line = line[start_column - 1:]

        method_body += line + "\n"

        brace_count += line.count("{")
        if(brace_count>0):
            found = True
        brace_count -= line.count("}")

        if brace_count == 0 and found:
            break

    return method_body

def extract_method_bodies_from_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()

    tree = javalang.parse.parse(java_code)

    package_name = ""
    class_name = ""

    for path, node in tree:
        if isinstance(node, javalang.tree.PackageDeclaration):
            package_name = ".".join(node.name)

        if isinstance(node, javalang.tree.ClassDeclaration):
            class_name = node.name

    method_info_list = []
    package_name=package_name.replace('...', '$')
    package_name=package_name.replace('.', '')
    package_name=package_name.replace('$', '.')
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        if node.body:
            start_line, start_column = node.position
            method_name = get_full_method_name(node, package_name, class_name)
            method_name=method_name.replace('$', '.')
            method_body = extract_method_body(java_code, start_line, start_column)
            method_info_list.append((method_name, method_body))

    return method_info_list

def extract_from_directory(directory, classList, methodList, system_path):
    extracted_methods=set()
    with open(system_path+"/method_with_body.csv", 'w', newline='') as file:
        csv_writer = csv.writer(file)
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    method_info_list = extract_method_bodies_from_file(file_path)
                    for method_info in method_info_list:
                        csv_writer.writerow([method_info[0], file, method_info[1]])
                        extracted_methods.add(method_info[0])  
                        
        # create all default constroctors
        for m in classList:
            const_name=m+"."+m.split(".")[-1]
            if const_name not in extracted_methods  :
                print(const_name)

                if const_name in methodList:
                    print("=======")

                    cons=createConstructor(m.split(".")[-1])
                    csv_writer.writerow([const_name, m.split(".")[-1]+'.java',cons])

def custom_tokenize(code, JAVA_STOP_WORDS):
    # Split camel case identifiers
    tokens = re.findall(r'\b\w+\b|[A-Z][a-z]*', code)
    # Process tokens to split underscores and combined camel case identifiers
    split_tokens = []
    for token in tokens:
        if '_' in token:
            # Split underscores into separate tokens
            split_tokens.extend(token.split('_'))
        else:
            # Split combined camel case identifiers further
            split_tokens.extend(re.findall(r'[A-Z][a-z]*|[a-z]+', token))
        # Remove numbers at the start or end of each token
    split_tokens = [re.sub(r'^\d+|\d+$', '', t) for t in split_tokens]

    # Remove empty tokens
    split_tokens = [t for t in split_tokens if t]

    # Initialize the Porter Stemmer and WordNet Lemmatizer
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Apply stemming and lemmatization
    stemmed_tokens = [porter_stemmer.stem(token) for token in split_tokens]
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in stemmed_tokens]

    filtered_tokens = [token for token in split_tokens if len(token) > 1]

    # Remove stop words
    filtered_tokens = [token for token in filtered_tokens if token.lower() not in JAVA_STOP_WORDS]

    return filtered_tokens

### Create Class Embaddings
def extract_class_bodies_from_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()

    tree = javalang.parse.parse(java_code)

    package_name = ""
    class_bodies = []

    for path, node in tree:
        if isinstance(node, javalang.tree.PackageDeclaration):
            package_name = ".".join(node.name)
        
        if isinstance(node, javalang.tree.ClassDeclaration):
            package_name=package_name.replace('...', '$')
            package_name=package_name.replace('.', '')
            package_name=package_name.replace('$', '.')
            class_name = node.name
            class_body = extract_class_body(java_code, node)
            class_bodies.append((package_name, class_name, class_body))
        

    return class_bodies

def extract_class_body(java_code, class_node):
    if class_node.position:
        start_line, _ = class_node.position
        lines = java_code.split('\n')
        # Find the opening brace of the class
        brace_found = False
        brace_count = 0
        for i, line in enumerate(lines[start_line-1:]):
            if '{' in line:
                brace_found = True
            if brace_found:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                if brace_count == 0:
                    # Class body found
                    return '\n'.join(lines[start_line-1:start_line+i])
    return ""

def extract_classes_from_directory(directory, system_path):
    class_with_body=dict()
    with open(system_path+"/class_with_body.csv", 'w', newline='') as file:
        csv_writer = csv.writer(file)        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    class_bodies = extract_class_bodies_from_file(file_path)
                    for c in class_bodies:
                        class_with_body[c[0]+'.'+c[1]]=c[2]
                        csv_writer.writerow([c[0]+'.'+c[1], c[2]])
    return class_with_body

def identify_domain_specific_keywords(tokens):
    word_count = Counter(tokens)

    # Use a percentile to set the threshold (e.g., 75th percentile)
    counts = np.array(list(word_count.values()))
    threshold = np.percentile(counts, 90)

    potential_keywords = [word for word, count in word_count.items() if count >= threshold]
    return potential_keywords


def create_embeddings(system_path, system_code_path):
    ### Prepare method list
    addedmethodSet = set()
    with open(system_path+'/classes_calls.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            addedmethodSet.add(row[0])
            addedmethodSet.add(row[1])

    methodList = set()
    line_count = 0
    with open(system_path+'/method_call_graph.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            line_count+=1
            methodList.add(row[0])
            methodList.add(row[1])

    classList = set()
    with open(system_path+'/classesList.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file,)
        for row in csv_reader:
            classList.add(row[0])

    extract_from_directory(system_code_path, classList, methodList, system_path)

    #### word2Vec
    JAVA_STOP_WORDS = [
        # Java keywords
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
        "class", "const", "continue", "default", "do", "double", "else", "enum",
        "extends", "final", "finally", "float", "for", "if", "implements", "import",
        "instanceof", "int", "interface", "long", "native", "new", "null", "package",
        "private", "protected", "public", "return", "short", "static", "strictfp",
        "super", "switch", "synchronized", "this", "throw", "throws", "transient",
        "try", "void", "volatile", "while", "test",
        
        # Common variable and method names
        "args", "array", "list", "map", "set", "string", "temp", "value",
        "index", "length", "size", "count", "result", "output", "input",
        "current", "previous", "next", "first", "last",
        "iterator", "collection", "object", "instance", "class", "variable", "constant",
        "parameter", "argument", "method", "function", "return", "exception", "error","test","csv"
    ,"test","from","java" ,"to","file",'maven', 'wrapper',
        # Common English words with little code-specific meaning
        "the", "and", "or", "not", "if", "else", "for", "while", "do", "in", "out",
        "print", "println", "printf", "format", "concat", "append", "substring", "length",
        "initialize", "iterate", "process", "calculate", "perform", "execute", "handle",
        "process", "implement", "generate", "construct", "create", "modify", "change",
        "update", "delete", "remove", "add", "insert", "retrieve", "fetch",
        
        # Common utility classes and methods
        "system", "out", "err", "printStackTrace", "println", "printf", "format",
        
        # Common programming terms
        "algorithm", "data", "structure", "implementation", "variable", "type", "instance",
        "object", "stack", "queue", "linkedlist","binary", "tree", "node", "element", "index", "size", "length", "head", "tail",
        "previous", "next", "first", "last", "current", "initialize", "iterate", "iterate"
        , 'mock', 'mvc', 'bean', 'repositori' , 'befor' , 'each', 'setup',
        'given', 'thi' ,'this', 'will', 'init', 'creation', 
        'form', 'except', 'get', 'expect', 'statu', 'is', 'ok', 'view', 'creat' 
        , 'updat' , 'model', 'attribut', 'exist', 'success', 'post', 
        'param', 'redirect', 'edit', 'ha', 'no', 'field', 'code', 'requir','json','action','controller',"alert","Action"
        
    ]

    # Download the 'punkt' and 'wordnet' resources
    nltk.download('punkt')
    nltk.download('wordnet')

    class_with_body=extract_classes_from_directory(system_code_path, system_path)

    allmethodNum=0
    methodListwithBody=set()
    usedMethods=set()
    tokenList={}
    embadList={}
    allTokens=[]
    tokenListBody={}
    with open(system_path+'/method_with_body.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            tokens=custom_tokenize(row[0].split('.')[-1], JAVA_STOP_WORDS)
            tokenList[row[0]]=tokens
            allTokens=allTokens+tokens
            
    for c in class_with_body:
        tokens=custom_tokenize(c.split('.')[-1], JAVA_STOP_WORDS)
        tokenListBody[c]=tokens
        allTokens=allTokens+tokens
        if (c.split('.')[-2] ):
            tokens=custom_tokenize(c.split('.')[-2], JAVA_STOP_WORDS)
            tokenListBody[c]+=tokens
            allTokens=allTokens+tokens
    potential_keywords= identify_domain_specific_keywords(allTokens)   

    model = Word2Vec(sentences=[allTokens], vector_size=100, window=5, min_count=1, workers=4)

    with open(system_path+'/method_embadding_w2v_withClass.csv', mode='w', newline='') as embadfile:
        embadfile_writer = csv.writer(embadfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for elem in tokenList:
            vectors_1 = [model.wv[token] for token in tokenList[elem] if token in model.wv]
            className='.'.join(elem.split('.')[:-1])
            mean_vector_1 = np.mean(vectors_1, axis=0)
            if className in tokenListBody :
                vectors_2 = [model.wv[token] for token in tokenListBody[className] if token in model.wv]
                if(not vectors_1 and vectors_2):
                    mean_vector_2 = np.mean(vectors_2, axis=0)
                    
                    if elem in methodList:
                        embadList[elem]=mean_vector_2
                        embadfile_writer.writerow([elem,mean_vector_2.tolist()])
                        usedMethods.add(elem)
                elif( vectors_1 and not vectors_2):
                    
                    if elem in methodList:
                        embadList[elem]=mean_vector_1
                        embadfile_writer.writerow([elem,mean_vector_1.tolist()])
                        usedMethods.add(elem)
                else:   
                    mean_vector_2 = np.mean(vectors_2, axis=0)
                    weighted_mean = (mean_vector_1 * 0.3) + (mean_vector_2 * 0.7)
                    
                    if elem in methodList:
                        embadList[elem]=weighted_mean
                        embadfile_writer.writerow([elem,weighted_mean.tolist()])
                        usedMethods.add(elem)
            else:
                
                if elem in methodList:
                    embadList[elem]=mean_vector_1
                    embadfile_writer.writerow([elem,mean_vector_1.tolist()])
                    usedMethods.add(elem)

    data = [embadList[method] for method in embadList]

    num_clusters=50
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)

    # Get cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Group methods by their cluster
    clustered_methods = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(labels):
        method_name = list(embadList.keys())[i]
        clustered_methods[label].append(method_name)

    # Print methods in each cluster
    print("\nMethods in Each Cluster:")
    for cluster in clustered_methods:
        print(f"Cluster {cluster}")
        for c in clustered_methods[cluster]:
            print(c)

    ### Create final call graph with methods from code
    with open(system_path+'/method_call_graph.csv', mode='r') as csv_file,open(system_path+'/call_graph.csv', mode='w', newline='') as graph_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_writer = csv.writer(graph_file, delimiter=';')
        for row in csv_reader:
            if(row[0] in usedMethods and row[1] in usedMethods):
                csv_writer.writerow([row[0],row[1]])



# curr_dir = r'C:\Users\oussa\OneDrive\Desktop\UdeM\PhD\others\imen\google-research-master\graph_embedding\dmon'
# system_path = curr_dir+"./data/POS2"
# system_code_path = curr_dir+'./data/POS2/JavaFX-Point-of-Sales-master'

# create_embeddings(system_path, system_code_path)


