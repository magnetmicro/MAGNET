import csv
from collections import defaultdict
import javalang

def calculate_smq(call_graph_path, microservice_map):
    # Load the call graph
    call_graph = defaultdict(set)
    with open(call_graph_path, 'r') as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            method1, method2 = row
            call_graph[method1].add(method2)

    # Calculate the number of methods for each microservice to represent the number of entities
    microservice_entities = defaultdict(int)
    for method in microservice_map.keys():
        microservice = microservice_map[method]
        microservice_entities[microservice] += 1

    # Calculate scoh for each microservice
    scoh = defaultdict(int)
    for method, connected_methods in call_graph.items():
        microservice = microservice_map.get(method)
        if microservice:
            for connected_method in connected_methods:
                if microservice_map.get(connected_method) == microservice:
                    scoh[microservice] += 1  # Increase edge count for the microservice

    # Calculate the final scoh values
    for microservice, num_methods in microservice_entities.items():
        scoh[microservice] = scoh[microservice] / (num_methods * (num_methods - 1) if num_methods > 1 else 1)

    # Calculate scop between microservices
    scop = defaultdict(lambda: defaultdict(int))
    for method, connected_methods in call_graph.items():
        for connected_method in connected_methods:
            service_i = microservice_map.get(method)
            service_j = microservice_map.get(connected_method)
            if service_i and service_j and service_i != service_j:
                scop[service_i][service_j] = 1

    # Calculate the final scop values
    for service_i, connections in scop.items():
        for service_j, edge_count in connections.items():
            scop[service_i][service_j] =  scop[service_i][service_j] /  (2 * microservice_entities[service_i] * microservice_entities[service_j])

    # Calculate SMQ
    N = len(microservice_entities)  # Number of microservices
    smq = sum(scoh.values()) / N
    # print(f"Number of clusters {N}")
    for service_i in microservice_entities:
        for service_j in microservice_entities:
            if service_i != service_j:
                smq -= (scop[service_i][service_j]/(2 * N * (N - 1) if N > 1 else 1))

    return smq
def entities_are_connected(entity1, entity2):
    return set(entity1.split()).intersection(set(entity2.split()))

def calculate_cmq(call_graph_path, method_bodies_path, microservice_map):
    # Load the call graph
    call_graph = defaultdict(set)
    with open(call_graph_path, 'r') as file:
        reader = csv.reader(file,delimiter=";")
        for row in reader:
            method1, method2 = row
            call_graph[method1].add(method2)

    # Load the method bodies to have access to the textual content
    method_bodies = {}
    with open(method_bodies_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            method = row[0]
            body= row[2]
            method_bodies[method] = body

    # Calculate ccoh for each microservice
    ccoh = defaultdict(int)
    for method, connected_methods in call_graph.items():
        microservice = microservice_map.get(method)
        if microservice:
            for connected_method in connected_methods:
                if microservice_map.get(connected_method) == microservice:
                    if entities_are_connected(method_bodies[method], method_bodies[connected_method]):
                        ccoh[microservice] += 1  # Increase edge count for the microservice

    # Calculate the number of methods (entities) for each microservice
    microservice_entities = defaultdict(int)
    for method in microservice_map.keys():
        microservice = microservice_map[method]
        microservice_entities[microservice] += 1

    # Calculate the final ccoh values
    for microservice, num_methods in microservice_entities.items():
        ccoh[microservice] = ccoh[microservice] / (num_methods**2) if num_methods > 1 else 0

    # Calculate ccop between microservices
    ccop = defaultdict(lambda: defaultdict(int))
    for method, connected_methods in call_graph.items():
        for connected_method in connected_methods:
            service_i = microservice_map.get(method)
            service_j = microservice_map.get(connected_method)
            if service_i and service_j and service_i != service_j:
                if entities_are_connected(method_bodies[method], method_bodies[connected_method]):
                    ccop[service_i][service_j] = 1

    # Calculate the final ccop values
    for service_i, connections in ccop.items():
        for service_j, edge_count in connections.items():
            ccop[service_i][service_j] = ccop[service_i][service_j] / (2 * microservice_entities[service_i] * microservice_entities[service_j])

    # Calculate CMQ
    N = len(microservice_entities)  # Number of microservices
    cmq = sum(ccoh.values()) / N

    for service_i in microservice_entities:
        for service_j in microservice_entities:
            if service_i != service_j:
                cmq -= ccop[service_i][service_j]
    cmq /= (N * (N - 1) / 2 if N > 1 else 1)

    return cmq
from itertools import combinations

# Placeholder function to extract domain terms from method names or bodies.
def f_term(method_name):
    # Extract domain terms from the method name.
    return set(method_name.split('.'))

def f_dom(opr_k, opr_m, operations_terms):
    term_k = operations_terms[opr_k]
    term_m = operations_terms[opr_m]
    return len(term_k.intersection(term_m)) / len(term_k.union(term_m)) if term_k.union(term_m) else 1

def calculate_chd(microservice_map):
    # Group methods by microservice
    microservices_methods = defaultdict(list)
    for method, service_id in microservice_map.items():
        microservices_methods[service_id].append(method)

    # Extract domain terms for each method
    operations_terms = {method: f_term(method) for method in microservice_map}

    # Calculate chd for each microservice
    chd_values = {}
    for service_id, methods in microservices_methods.items():
        if len(methods) > 1:
            chd_sum = sum(f_dom(opr_k, opr_m, operations_terms) for opr_k, opr_m in combinations(methods, 2))
            chd_j = 2*chd_sum / (len(methods) * (len(methods) -1))
            chd_values[service_id] = chd_j
        else:
            chd_values[service_id] = 1  # Default value if there is only one operation in the microservice

    # Compute CHD which is the average of all chd values
    CHD = sum(chd_values.values()) / len(chd_values) if chd_values else 0
    return CHD, chd_values


def extract_parameters_and_return(method_body):
    # Wrap the method body in a class structure
    wrapped_method_body = f"class DummyClass {{ {method_body} }}"

    try:
        # Parse the wrapped method body into an AST
        tree = javalang.parse.parse(wrapped_method_body)
    except javalang.parser.JavaSyntaxError as e:
        # print(f"----------Syntax error in source code: {e}")
        # print(method_body)

        return set(), set()

    # Extracting the first method declaration in the AST
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        return_type = node.return_type.name if node.return_type else None
        return_types = {return_type} if return_type else set()

        # Extract parameter types, considering annotations
        parameter_types = set()
        for param in node.parameters:
            # Check if parameter is annotated
            if isinstance(param.type, javalang.tree.ReferenceType):
                param_type = param.type.name
            elif isinstance(param.type, javalang.tree.Annotation):
                param_type = param.type.element.name
            else:
                param_type = None

            if param_type:
                parameter_types.add(param_type)

        return return_types, parameter_types

    return set(), set()

def f_msg(opr_k, opr_m, operations_details):
    ret_k, param_k = operations_details[opr_k]
    ret_m, param_m = operations_details[opr_m]
    similarity_ret = len(ret_k.intersection(ret_m)) / len(ret_k.union(ret_m)) if ret_k.union(ret_m) else 1
    similarity_param = len(param_k.intersection(param_m)) / len(param_k.union(param_m)) if param_k.union(param_m) else 1
    return (similarity_ret + similarity_param) / 2

def calculate_chm(method_bodies_path, microservice_map):
    # Reverse the microservice_map to group methods by microservice
    microservices = defaultdict(list)
    for method, service_id in microservice_map.items():
        microservices[service_id].append(method)

    # Load the method bodies and extract input parameters and return values
    operations_details = {}
    with open(method_bodies_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            method = row[0]
            body= row[2]
            operations_details[method] = extract_parameters_and_return(body)
    # Calculate chm for each microservice
    chm_values = defaultdict(float)

    for service_id, methods in microservices.items():
        if len(methods) > 1:
            pairs = combinations(methods, 2)
            chm_sum = sum(f_msg(opr_k, opr_m, operations_details) for opr_k, opr_m in pairs)
            chm_values[service_id] = chm_sum / (len(methods) * (len(methods) - 1) / 2)
        else:
            chm_values[service_id] = 1  # Default value if there is only one method in the microservice

    # Compute CHM which is the average of all chm values for the microservices with more than one method
    CHM = sum(chm_values.values()) / len(chm_values)
    return CHM, dict(chm_values)
