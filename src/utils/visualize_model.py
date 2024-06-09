from torchview import draw_graph
from graphviz.graphs import Digraph
from torchvision.models.segmentation.deeplabv3 import DeepLabV3

def generate_graph(model: DeepLabV3, destination: str) -> Digraph:
    """
    Creates a graph to visualize the architecture of the DeeplabV3 model.

    Parameters
    ----------
    model: DeeplabV3
        The DeepLab3 model to be visualized.

    destination: str
        The path where the generated graph will be saved.

    Returns
    -------
    graoh: Digraph
        A digraph object that visualizes the model.
    """

    model_graph = draw_graph(
        model, input_size=(1, 1, 512, 512),
        graph_name="DeepLabV3",
        expand_nested=True,
        save_graph=True, directory=destination,
        filename="DeepLabV3-architecture"
    )

    graph = model_graph.visual_graph
    
    return graph