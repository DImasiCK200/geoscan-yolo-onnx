import onnx
import onnx_graphsurgeon as gs

import numpy as np
import copy

import argparse
from pathlib import Path


def updateDicts(graph):

    outputsDict = {}
    inputsDict = {}
    namesDict = {}

    for index in range(len(graph.nodes)):
        
        namesDict[graph.nodes[index].name] = index

        for output in graph.nodes[index].outputs:
            outputsDict[output.name] = index
        
        for inpt in graph.nodes[index].inputs:
            inputsDict[inpt.name] = index
    
    return outputsDict, inputsDict, namesDict


def outToIdx(outputs):
    global outputsDict
    
    outputsIndex = []
    for output in outputs:
        outputsIndex.append(outputsDict[output])
    return outputsIndex


def searchElem(graph, start, opType, maxDepth=7):
    global outputsDict

    if not maxDepth:
        return []
    
    node = graph.nodes[start]
    if node.op == opType:
        outputs =[]
        for output in node.outputs:
            outputs.append(output.name)
        return outputs
    
    inputs = []
    for inPut in node.inputs:
        if inPut.name in outputsDict:
            inputs.extend(searchElem(graph, outputsDict[inPut.name], opType, maxDepth - 1))
    
    return set(inputs)


def getOutput(graph, convIdxs):
    
    outputs = []

    for i in convIdxs:
        outputs.extend(graph.nodes[i].outputs)
    
    return outputs


def delNodes(graph, nodesToDel):
    
    nodesToDel = list(nodesToDel)
    nodesToDel.sort(reverse=True)

    for i in nodesToDel:
        del graph.nodes[i]

    graph.cleanup()


def getNewName(name='', preffix=''):
    global counts

    if name not in counts:
        counts[name] = 0

    counts[name] += 1

    if preffix:
        return f"/{preffix}/{name}_{counts[name]}"
    
    if name:
        return f"{name}_{counts[name]}"
    
    return f"{counts[name]}"


def getClipInputs():

    arr = []
    
    for i in [0, 1]:
        arr.append(gs.Constant(
                name=getNewName("Constant"),
                values=np.array(i, dtype=np.float32)
            ))
    
    return arr


def createOutput(opType='', graphOutput=False, name='', dtype="float32", shape=None):
    outputBase = {
        "dtype" : dtype, 
        "shape" : shape,
    }
    global preffix

    if opType:
        outputBase["name"] = getNewName(opType, preffix)
        return gs.Variable(**outputBase)

    if name:
        outputBase["name"] = getNewName(name)
        return gs.Variable(**outputBase)
    
    outputBase["name"] = getNewName()
    
    return gs.Variable(**outputBase)


def createNode(graph, newNodesInfo, inputs, shapeEnd=None):

    if newNodesInfo:
        nodeInfo = newNodesInfo.pop(0)
        outputs = copy.deepcopy(nodeInfo["outputs"][0])

        if nodeInfo["inputsConst"]:
            inputs.extend(getClipInputs())
        
        if outputs["shape"]:
            outputs["shape"].extend([shapeEnd] * 2)
        
        output = [createOutput(**outputs)]

        graph.outputs.extend(output * outputs["graphOutput"])

        nodeBase = {
            "op" : nodeInfo["opType"],
            "name" : getNewName(nodeInfo["opType"]),
            "inputs" : inputs,
            "outputs" : output,
        }

        if nodeInfo["attrs"]:
            nodeBase['attrs'] = nodeInfo["attrs"]
        
        node = gs.Node(**nodeBase)
            
        graph.nodes.append(node)

        createNode(graph, newNodesInfo, output, shapeEnd)
    

newElements = [
    {
        "opType" : 'Sigmoid',
        "attrs" : None,
        "inputsConst": False,
        "outputs" : [
            {
                "name" : "onnx::ReduceSum",
                "graphOutput" : True,
                "shape" : [1, 80],
                "dtype" : "float32",
            }
        ]
    },
    {
        "opType" : 'ReduceSum',
        "attrs" : {
            "axes": [1],
            "keepdims": 1,
        },
        "inputsConst": False,
        "outputs" : [
            {
                "opType" : 'ReduceSum',
                "graphOutput" : False,
                "shape" : None,
                "dtype" : None,
            }
        ]
    },
    {
        "opType" : 'Clip',
        "attrs" : None,
        "inputsConst": True,
        "outputs" : [
            {
                "graphOutput" : True,
                "shape" : [1, 1],
                "dtype" : "float32",
            }
        ]
    }
]

counts = {}


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Путь к ONNX модели")
parser.add_argument("-o", "--out", help="Путь для сохранения новой модели")
args = parser.parse_args()

modelPath = Path(args.model)

if not modelPath.exists():
    raise SystemExit(f"Проверьте введеный путь. Файл не найден.")

if args.out:
    newModelPath = Path(args.out)

else:
    newModelPath = modelPath.with_name(modelPath.stem + "_mod.onnx")

print(f'Загрузка модели.. {modelPath}')

model = model = onnx.load(modelPath)

graph = gs.import_onnx(model)

outputsDict, inputsDict, namesDict = updateDicts(graph)


convOuts = searchElem(graph, -1, 'Conv')
convOutIdxs = outToIdx(convOuts)


nodesToDel = set()
for i in convOuts:
    nodesToDel.add(inputsDict[i])


newOutputs = getOutput(graph, convOutIdxs)
graph.outputs = []

for idx in convOutIdxs:
    output = graph.nodes[idx].outputs[0]
    preffix = output.name.strip("/").split("/")[0]
    shape = output.shape

    if shape[1] == 64:
        newOutput = createOutput(shape=shape)
        graph.nodes[idx].outputs.append(newOutput)
        graph.outputs.append(newOutput)

    else:
        createNode(graph, newElements.copy(), [output], shapeEnd=shape[-1])


delNodes(graph, nodesToDel)

print(f'Сохранение обработанной модели.. {newModelPath}')
onnx.save(gs.export_onnx(graph), newModelPath)
print('Модель сохранена')
