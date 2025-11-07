import onnx
import onnx_graphsurgeon as gs
import numpy as np
import copy
import argparse
from pathlib import Path


class OnnxGraphModifier:
    def __init__(self, model_path: Path, output_path: Path = None):
        self.model_path = model_path
        self.new_model_path = output_path or model_path.with_name(model_path.stem + "_mod.onnx")

        self.counts = {}
        self.outputsDict = {}
        self.inputsDict = {}
        self.namesDict = {}
        self.preffix = ""

    # --------------------- служебные методы ----------------------

    def updateDicts(self, graph):
        self.outputsDict.clear()
        self.inputsDict.clear()
        self.namesDict.clear()

        for index, node in enumerate(graph.nodes):
            self.namesDict[node.name] = index
            for output in node.outputs:
                self.outputsDict[output.name] = index
            for inpt in node.inputs:
                self.inputsDict[inpt.name] = index

    def outToIdx(self, outputs):
        return [self.outputsDict[o] for o in outputs]

    def searchElem(self, graph, start, opType, maxDepth=7):
        if not maxDepth:
            return []

        node = graph.nodes[start]
        if node.op == opType:
            return [out.name for out in node.outputs]

        inputs = []
        for inPut in node.inputs:
            if inPut.name in self.outputsDict:
                inputs.extend(self.searchElem(graph, self.outputsDict[inPut.name], opType, maxDepth - 1))
        return set(inputs)

    def getOutput(self, graph, convIdxs):
        outputs = []
        for i in convIdxs:
            outputs.extend(graph.nodes[i].outputs)
        return outputs

    def delNodes(self, graph, nodesToDel):
        nodesToDel = sorted(list(nodesToDel), reverse=True)
        for i in nodesToDel:
            del graph.nodes[i]
        graph.cleanup()

    def getNewName(self, name='', preffix=''):
        if name not in self.counts:
            self.counts[name] = 0
        self.counts[name] += 1

        if preffix:
            return f"/{preffix}/{name}_{self.counts[name]}"
        return f"{name}_{self.counts[name]}"

    def getClipInputs(self):
        arr = []
        for i in [0, 1]:
            arr.append(gs.Constant(
                name=self.getNewName("Constant"),
                values=np.array(i, dtype=np.float32)
            ))
        return arr

    def createOutput(self, opType='', graphOutput=False, name='', dtype="float32", shape=None):
        outputBase = {"dtype": dtype, "shape": shape}

        if opType:
            outputBase["name"] = self.getNewName(opType, self.preffix)
            return gs.Variable(**outputBase)

        if name:
            outputBase["name"] = self.getNewName(name)
            return gs.Variable(**outputBase)

        outputBase["name"] = self.getNewName()
        return gs.Variable(**outputBase)

    def createNode(self, graph, newNodesInfo, inputs, shapeEnd=None):
        if newNodesInfo:
            nodeInfo = newNodesInfo.pop(0)
            outputs = copy.deepcopy(nodeInfo["outputs"][0])

            if nodeInfo["inputsConst"]:
                inputs.extend(self.getClipInputs())

            if outputs["shape"]:
                outputs["shape"].extend([shapeEnd] * 2)

            output = [self.createOutput(**outputs)]
            graph.outputs.extend(output * outputs["graphOutput"])

            nodeBase = {
                "op": nodeInfo["opType"],
                "name": self.getNewName(nodeInfo["opType"]),
                "inputs": inputs,
                "outputs": output,
            }

            if nodeInfo["attrs"]:
                nodeBase['attrs'] = nodeInfo["attrs"]

            node = gs.Node(**nodeBase)
            graph.nodes.append(node)

            self.createNode(graph, newNodesInfo, output, shapeEnd)

    # --------------------- основной процесс ----------------------

    def process(self):
        print(f'Загрузка модели.. {self.model_path}')
        model = onnx.load(self.model_path)
        graph = gs.import_onnx(model)

        self.updateDicts(graph)

        convOuts = self.searchElem(graph, -1, 'Conv')
        convOutIdxs = self.outToIdx(convOuts)

        nodesToDel = {self.inputsDict[i] for i in convOuts}
        newOutputs = self.getOutput(graph, convOutIdxs)
        graph.outputs = []

        newElements = [
            {
                "opType": 'Sigmoid',
                "attrs": None,
                "inputsConst": False,
                "outputs": [
                    {
                        "name": "onnx::ReduceSum",
                        "graphOutput": True,
                        "shape": [1, 80],
                        "dtype": "float32",
                    }
                ]
            },
            {
                "opType": 'ReduceSum',
                "attrs": {"axes": [1], "keepdims": 1},
                "inputsConst": False,
                "outputs": [
                    {
                        "opType": 'ReduceSum',
                        "graphOutput": False,
                        "shape": None,
                        "dtype": "float32",
                    }
                ]
            },
            {
                "opType": 'Clip',
                "attrs": None,
                "inputsConst": True,
                "outputs": [
                    {"graphOutput": True, "shape": [1, 1], "dtype": "float32"}
                ]
            }
        ]

        for output in newOutputs:
            self.preffix = output.name.strip("/").split("/")[0]
            shape = output.shape
            if shape[1] == 64:
                graph.outputs.append(output)
            else:
                self.createNode(graph, newElements.copy(), [output], shapeEnd=shape[-1])

        self.delNodes(graph, nodesToDel)

        print(f'Сохранение обработанной модели.. {self.new_model_path}')
        onnx.save(gs.export_onnx(graph), self.new_model_path)
        print('Модель сохранена')


# --------------------- запуск ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Путь к ONNX модели")
    parser.add_argument("-o", "--out", help="Путь для сохранения новой модели")
    args = parser.parse_args()

    modelPath = Path(args.model)
    if not modelPath.exists():
        raise SystemExit("Ошибка: файл модели не найден.")

    modifier = OnnxGraphModifier(modelPath, Path(args.out) if args.out else None)
    modifier.process()
