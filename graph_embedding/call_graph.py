import os
from lxml import etree as ET
import csv


# Get all classes of the system
def find_java_files(directory):
    java_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def extract_package_names(java_file):
    package_names = []
    with open(java_file, 'r',errors='ignore') as file:
        for line in file:
            if line.strip().startswith("package "):
                package_names.append(line.strip().split()[1][:-1])
    package_names.append(java_file.split("/")[-1].split(".")[0])
    return package_names

def get_method_by_path(root, path, namespaces):
    parts = path.split("/")
    current_element = root  #  the root is <kdm:Segment> or <xmi:XMI>
    fullName=""
    # Skipping the first split because it's always 0 which means the root
    for part in parts[1:]:
        part=part.strip("@")
        # If part doesn't contain model or codeElement, it's not valid
        if ('codeElement.' not in part) and ('model.'  not in part):
            continue

        # Get element type and index
        children = []
        element_type, index_str = part.split('.')
        index = int(index_str)   
        if element_type == 'codeElement':
            children = current_element.findall('./codeElement', namespaces)
            current_element=children[index]
        elif element_type == 'model':
            children = current_element.findall('.//model', namespaces)  
            current_element=children[index]
            continue
            
        if(current_element.get('name') is not None):
            fullName=fullName+current_element.get('name')+"."           
        if current_element.get(f"{{{namespaces['xsi']}}}type") == "code:MethodUnit":
            return fullName.strip(".")
        
    return None

def get_method_by_path_construct(root, path, namespaces):
    parts = path.split("/")
    current_element = root  #  the root is <kdm:Segment> or <xmi:XMI>
    fullName=""
    # Skipping the first split because it's always 0 which means the root
    for part in parts[1:]:
        part=part.strip("@")
        # If part doesn't contain model or codeElement, it's not valid
        if ('codeElement.' not in part) and ('model.'  not in part):
            continue

        # Get element type and index
        children = []
        element_type, index_str = part.split('.')
        index = int(index_str)   
        if element_type == 'codeElement':
            children = current_element.findall('./codeElement', namespaces)
            current_element=children[index]
        elif element_type == 'model':
            children = current_element.findall('.//model', namespaces)  
            current_element=children[index]
            continue            
        if(current_element.get('name') is not None):
            fullName=fullName+current_element.get('name')+"."           
        if current_element.get(f"{{{namespaces['xsi']}}}type") == "code:ClassUnit":
            fullName=fullName+current_element.get('name')
            return fullName
        
    return None

def create_call_graph(kdm_file_path,csv_file_path, namespaces):
    tree = ET.parse(kdm_file_path)
    root = tree.getroot()
    methods = root.findall(".//*[@xsi:type='code:MethodUnit']", namespaces=namespaces)
    calls = root.findall(".//*[@xsi:type='action:Calls']", namespaces=namespaces)
    creates = root.findall(".//*[@xsi:type='action:Creates']", namespaces=namespaces)
   
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for call in calls:
            to_reference = call.attrib['to']
            from_reference = call.attrib['from']
            to_element=get_method_by_path(root,to_reference,namespaces)
            from_element=get_method_by_path(root,from_reference,namespaces)
            if to_element is not None and from_element is not None:
                writer.writerow([from_element, to_element])
        for createCall in creates:
            to_reference = createCall.attrib['to']
            from_reference = createCall.attrib['from']
            to_element=get_method_by_path(root,to_reference,namespaces)
            from_element=get_method_by_path_construct(root,from_reference,namespaces)
            if to_element is not None and from_element is not None:
                writer.writerow([from_element, to_element])

def get_class_by_path(root, path, namespaces):
    parts = path.split("/")
    current_element = root  #  the root is <kdm:Segment> or <xmi:XMI>
    fullName=""
    # Skipping the first split because it's always 0 which means the root
    for part in parts[1:]:
        part=part.strip("@")
        # If part doesn't contain model or codeElement, it's not valid
        if ('codeElement.' not in part) and ('model.'  not in part):
            continue

        # Get element type and index
        children = []
        element_type, index_str = part.split('.')
        index = int(index_str)   
        if element_type == 'codeElement':
            children = current_element.findall('./codeElement', namespaces)
            current_element=children[index]
        elif element_type == 'model':
            children = current_element.findall('.//model', namespaces)  
            current_element=children[index]
            continue
        if(current_element.get('name') is not None):
            fullName=fullName+current_element.get('name')+"."  
        #if current_element.get(f"{{{namespaces['xsi']}}}type") == "code:MethodUnit":
            #return fullName.strip(".")
        
    return  fullName.strip(".")

def get_classes_relations(root, relations, file, type_relation, classesList, namespaces, system_path):
    with open(file, 'a+', newline='') as csv_file, open(system_path+"/method_call_graph.csv", 'a+') as file2:
        writer = csv.writer(csv_file)
        writer2 = csv.writer(file2)
        for relation in relations:
            to_reference = relation.attrib['to']
            from_reference = relation.attrib['from']
            to_element=get_class_by_path(root,to_reference,namespaces)
            from_element=get_class_by_path(root,from_reference,namespaces)
            if to_element is not None and to_element in classesList and from_element is not None and from_element in classesList:
                writer.writerow([from_element, to_element, type_relation])
                print([from_element+"."+from_element.split('.')[-1], to_element+"."+to_element.split('.')[-1]])
                writer2.writerow([from_element+"."+from_element.split('.')[-1], to_element+"."+to_element.split('.')[-1]])



def generate_CG(system_path, kdm_file_path, system_code_path):
    # create_directory
    try:
        os.mkdir(system_path)
        print(f"Directory '{system_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{system_path}' already exists.")

    # Find all Java files in the project
    java_files = find_java_files(system_code_path)

    # Extract and print the full package names
    classe_file=system_path+'/classesList.csv'
    classesList=[]
    with open(classe_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for java_file in java_files:
            package_names = extract_package_names(java_file)
            if package_names:
                writer.writerow({'.'.join(package_names)})
                classesList.append('.'.join(package_names))

    root, ext = os.path.splitext(kdm_file_path)
    ext = ext.lower() 
    if not os.path.isfile(kdm_file_path):
        print("The KDM file is not found")
    if ext != '.xmi':
        print("The kdm should be an XMI file")
        
    if not os.path.isdir(system_code_path):
        print("The system source code is not found")

    ## Parse KDM to create a call graph
    namespaces = {
        'xmi': 'http://www.omg.org/XMI',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'action': 'http://www.eclipse.org/MoDisco/kdm/action',
        'code': 'http://www.eclipse.org/MoDisco/kdm/code',
        'kdm': 'http://www.eclipse.org/MoDisco/kdm/kdm',
        'source': 'http://www.eclipse.org/MoDisco/kdm/source'
    }

    csv_file_path = system_path+'/method_call_graph.csv'
    create_call_graph(kdm_file_path,csv_file_path, namespaces)

    ### Extract classes
    tree = ET.parse(kdm_file_path)
    root = tree.getroot()
    # add instantiate
    relations = root.findall(".//*[@xsi:type='code:Extends']", namespaces=namespaces)
    get_classes_relations(root, relations,system_path+'/classes_calls.csv','Extends', classesList, namespaces, system_path)
    relations = root.findall(".//*[@xsi:type='code:Implements']", namespaces=namespaces)
    get_classes_relations(root, relations,system_path+'/classes_calls.csv','Implements', classesList, namespaces, system_path)


# curr_dir = r'C:\Users\oussa\OneDrive\Desktop\UdeM\PhD\others\imen\google-research-master\graph_embedding\dmon'
# system_path = curr_dir+"./data/POS2"
# kdm_file_path = curr_dir+'./data/POS2/KDM/inventory_kdm.xmi'
# system_code_path = curr_dir+'./data/POS2/JavaFX-Point-of-Sales-master'

# generate_CG(system_path, kdm_file_path, system_code_path)