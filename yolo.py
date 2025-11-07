import onnx
import onnx_graphsurgeon as gs
from pathlib import Path
import numpy as np
import copy


# ==========================================================
# 🔧 Вспомогательные функции
# ==========================================================

def new_path(path: str) -> Path:
    """Создает путь к новому файлу с суффиксом '_modified'."""
    old_path = Path(path)
    return old_path.parent / f"{old_path.stem}_modified{old_path.suffix}"


def update_dicts(graph: gs.Graph):
    """Формирует словари для быстрого доступа по именам."""
    outputs_dict, inputs_dict, names_dict = {}, {}, {}
    for idx, node in enumerate(graph.nodes):
        names_dict[node.name] = idx
        outputs_dict.update({out.name: idx for out in node.outputs})
        inputs_dict.update({inp.name: idx for inp in node.inputs})
    return outputs_dict, inputs_dict, names_dict


def out_to_idx(outputs, outputs_dict):
    """Преобразует список имен выходов в список индексов."""
    return [outputs_dict[o] for o in outputs if o in outputs_dict]


def search_elem(graph, start, op_type, outputs_dict, max_depth=7):
    """Поиск выходов узлов с указанным типом (итеративно, без рекурсии)."""
    stack = [(start, 0)]
    results = set()

    while stack:
        idx, depth = stack.pop()
        if depth >= max_depth:
            continue

        node = graph.nodes[idx]
        if node.op == op_type:
            results.update(out.name for out in node.outputs)
            continue

        for inp in node.inputs:
            if inp.name in outputs_dict:
                stack.append((outputs_dict[inp.name], depth + 1))

    return results


def get_outputs(graph, node_indices):
    """Возвращает все выходы заданных узлов."""
    outputs = []
    for i in node_indices:
        outputs.extend(graph.nodes[i].outputs)
    return outputs


def del_nodes(graph, nodes_to_del):
    """Удаляет указанные узлы и очищает граф."""
    for i in sorted(nodes_to_del, reverse=True):
        del graph.nodes[i]
    graph.cleanup(remove_unused_graph_inputs=True, remove_unused_node_outputs=True)


def get_new_name(counts, name='', prefix=''):
    """Генерирует уникальные имена узлов и переменных."""
    counts[name] = counts.get(name, 0) + 1
    if prefix:
        return f"/{prefix}/{name}_{counts[name]}"
    return f"{name}_{counts[name]}"


def get_clip_inputs(counts):
    """Создает константы для узла Clip (min=0, max=1)."""
    return [
        gs.Constant(name=get_new_name(counts, "Constant"), values=np.array(0, dtype=np.float32)),
        gs.Constant(name=get_new_name(counts, "Constant"), values=np.array(1, dtype=np.float32))
    ]


def create_output(counts, prefix='', op_type='', name='', dtype="float32", shape=None):
    """Создает выходную переменную (Variable) для узла."""
    base = {"dtype": dtype, "shape": shape}

    if op_type:
        base["name"] = get_new_name(counts, op_type, prefix)
    elif name:
        base["name"] = get_new_name(counts, name)
    else:
        base["name"] = get_new_name(counts, "output")

    return gs.Variable(**base)


def create_nodes_chain(graph, new_nodes_info, inputs, counts, prefix, shape_end=None):
    """Создает последовательность узлов из списка new_nodes_info."""
    current_inputs = inputs

    for node_info in new_nodes_info:
        outputs_info = copy.deepcopy(node_info["outputs"][0])

        # Добавляем константы для Clip
        if node_info["inputsConst"]:
            current_inputs.extend(get_clip_inputs(counts))

        # Добавляем недостающие размеры, если указано shapeEnd
        if outputs_info.get("shape"):
            outputs_info["shape"].extend([shape_end] * 2)

        # Создание выхода
        output = [create_output(counts, prefix=prefix, **outputs_info)]

        # Добавляем выход в граф, если нужно
        if outputs_info.get("graphOutput"):
            graph.outputs.extend(output)

        # Создание узла
        node_base = {
            "op": node_info["opType"],
            "name": get_new_name(counts, node_info["opType"]),
            "inputs": current_inputs,
            "outputs": output,
        }

        if node_info["attrs"]:
            node_base['attrs'] = node_info["attrs"]

        node = gs.Node(**node_base)
        graph.nodes.append(node)

        # Следующий узел получает выход текущего
        current_inputs = output


# ==========================================================
# 🚀 Основная логика
# ==========================================================

def main():
    model_path = './models/yolov8n.onnx'
    new_model_path = new_path(model_path)

    print(f'Загрузка модели: {model_path}')
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)

    outputs_dict, inputs_dict, names_dict = update_dicts(graph)

    # Ищем выходы Conv-узлов
    conv_outputs = search_elem(graph, -1, 'Conv', outputs_dict)
    conv_output_idxs = out_to_idx(conv_outputs, outputs_dict)

    # Список узлов на удаление
    nodes_to_del = {inputs_dict[o] for o in conv_outputs if o in inputs_dict}

    # Новые выходы и очистка старых
    new_outputs = get_outputs(graph, conv_output_idxs)
    graph.outputs = []

    # Конфигурация новых добавляемых узлов
    new_elements = [
        {
            "opType": 'Sigmoid',
            "attrs": None,
            "inputsConst": False,
            "outputs": [{
                "name": "onnx::ReduceSum",
                "graphOutput": True,
                "shape": [1, 80],
                "dtype": "float32",
            }]
        },
        {
            "opType": 'ReduceSum',
            "attrs": {"axes": [1], "keepdims": 1},
            "inputsConst": False,
            "outputs": [{
                "opType": 'ReduceSum',
                "graphOutput": False,
                "shape": None,
                "dtype": "float32",
            }]
        },
        {
            "opType": 'Clip',
            "attrs": None,
            "inputsConst": True,
            "outputs": [{
                "graphOutput": True,
                "shape": [1, 1],
                "dtype": "float32",
            }]
        }
    ]

    counts = {}

    # Проходим по выходам Conv и добавляем цепочку узлов
    for output in new_outputs:
        prefix = output.name.strip("/").split("/")[0]
        shape = output.shape

        if not shape or len(shape) < 2:
            continue

        if shape[1] == 64:
            graph.outputs.append(output)
        else:
            create_nodes_chain(graph, copy.deepcopy(new_elements), [output], counts, prefix, shape_end=shape[-1])

    # Удаляем старые Conv-узлы
    del_nodes(graph, nodes_to_del)

    # Сохраняем результат
    print(f'Сохранение обработанной модели: {new_model_path}')
    onnx.save(gs.export_onnx(graph), new_model_path)
    print('✅ Модель успешно сохранена.')


# ==========================================================
# 📦 Точка входа
# ==========================================================

if __name__ == "__main__":
    main()
