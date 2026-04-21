# Blending Optimization of Networks with Uncertain Scenarios (BONUS)

This repository contains the code for modeling and optimizing gas + hydrogen pipeline networks, with a focus on the Norwegian hydrogen infrastructure. The project includes deterministic and stochastic optimization models, data analysis tools, visualization scripts, and experimental scenarios.

## Overview

The models optimize hydrogen and gas blending transportation networks considering factors such as pipeline capacities, compression costs, pressure constraints, and stochastic uncertainties in demand and supply. The project uses Pyomo for optimization, NetworkX for graph representation, and GeoPandas for geospatial analysis.

## Features

- **Network Modeling**: Creation and analysis of hydrogen pipeline networks
- **Optimization Models**: Deterministic and stochastic variants using Pyomo
- **Data Analysis**: Processing of geospatial data and network parameters
- **Visualization**: Interactive maps and graphs of network flows and results
- **Experiments**: Various scenarios including market experiments, subsidy impacts, and technical parameters

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hydrogen_gas
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Required packages:
- networkx
- geopandas
- matplotlib
- numpy
- pandas
- scipy
- pyomo
- openpyxl
- shapely
- folium
- pyyaml

## Usage

### Running the Main Model

Navigate to the `study_case_model/` directory and run the stochastic model from the main repo folder:

```bash
python study_case_stochastic_model.py
```

### Data Preparation

Use scripts in `data/data_prepare_functions/` to process network data:
- `create_bigger_network.py`
- `create_smaller_network.py`
- `hydrogen_map.py`

### Visualization

Run visualization scripts in `study_case_model/study_case_figures.py` or `data/data_analysis/html_graph.py` to generate maps and graphs.

## Project Structure

- `config.yaml`: Configuration file with paths
- `data/`: Data files and analysis scripts
  - `data_sources/`: Raw data (node info, generation fields, pipelines)
  - `data_analysis/`: Analysis and visualization scripts
  - `data_prepare_functions/`: Network creation and feature addition scripts
- `study_case_model/`: Main optimization model
  - `study_case_stochastic_model.py`: Stochastic optimization model
  - `study_case_problem_file.py`: Problem formulation and helper functions
  - `bigger_network/` and `smaller_network/`: Network-specific solvers
  - `Experiments/`: Experimental setups and results
  - `figures/`: Generated plots and maps
- `Old models/`: Legacy deterministic and stochastic models
- `scenario_variables/`: Experiment data and parameters
- `logs/`: Execution logs
- `nodefiles/`: Node data files

## Key Concepts

- **Stages and Branches**: Multi-stage stochastic programming with branching scenarios
- **Cutting Planes**: Approximation techniques for nonlinear flow equations (Weymouth equation)
- **Compression and Pressure**: Modeling of compression costs and pressure constraints
- **Homogeneous Flows**: Optimization of gas composition flows through pipelines

## Contributing

Please ensure all changes are tested and documented. Update the README if new features are added.

## License

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contact

E-mail: larson.beemster@gmail.com
Tel: +31 6 1942 1812