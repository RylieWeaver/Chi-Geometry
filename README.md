# Chi-Geometry
Welcome to **Chi-Geometry!** A repository to easily generate benchmark datasets for chirality prediction of GNNs.

Chi-Geometry generates chiral samples in a purely geometric and topological way, making it ideal for testing a GNNs ability to predict chirality without extraneous chemical correlations.

![Chiral Configurations](images/configurations_table.png)


## Quick Start

Download the library, generate a dataset, and visualize it in just 3 steps:

```bash
# 1. Install the Library
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple chi_geometry
```

```python
# 2. Generate the Dataset
num_samples = 50
chirality_type = "simple"
chirality_distance = 1
species_range = 15
noise = False
dataset = create_dataset(
    num_samples, chirality_type, chirality_distance, species_range, noise
)

# 3. Visualize dataset (optional)
plot_graph(dataset[0])
```


## Configuration Options

Chi-Geometry provides three main configuration options which are read from json:

### Chirality Type
This option defines the structural arrangement of the chiral configuration:
- **Simple**: A chiral center with three neighbor chains and corresponding connections.
- **Crossed**: Similar to the simple configuration, but with "crossed" connections between the neighbor chains to add complexity.
- **Classic**: Reflects how chirality typically appears in chemistry, with a chiral center connected to four neighbor chains.

### Distance
Specifies the hop distance between the chiral center and the nodes which determine its chiral tag. All intermediate layers consist of the same atom type, so the unique chiral configuration is only defined by the last node of each neighbor chain.

### Species Range
Sets the variety of unique node types (species) available for random selection within the chiral configuration.

### Noise
Introduce positional noise to mitigate chiral classification by extraneous positional information.


## Quick Example

Set up your environment, generate a dataset, and train an E(3)-equivariant model in just two steps:

```bash
# 1. Set up environment
git clone https://github.com/RylieWeaver/Chi-Geometry.git
cd Chi-Geometry
export PYTHONPATH=$(pwd)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple chi_geometry
cd examples
pip install model_requirements.txt

# 2. Run example
cd classic
python main.py
```
