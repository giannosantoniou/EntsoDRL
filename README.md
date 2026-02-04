# EntsoDRL: Deep Reinforcement Learning for Battery Trading

## Overview
This project implements a Deep Reinforcement Learning (DRL) system for automated battery energy trading in the Greek Energy Market (HEnEx). The agent interacts with a custom simulation environment to learn optimal charging/discharging strategies that maximize profit while minimizing battery degradation.

## Architecture
- **Environment**: Custom Gymnasium environment (`BatteryTradingEnv`) simulating battery physics and market dynamics.
- **Agent**: DRL agent (PPO/SAC) implemented using Stable Baselines 3.
- **Safety**: Hard constraints enforced via Action Masking.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd EntsoDRL
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Docker Deployment
1.  **Build Services**:
    ```bash
    docker-compose build
    ```
2.  **Run Training**:
    ```bash
    docker-compose up training
    ```
3.  **Run Evaluation**:
    ```bash
    docker-compose up evaluation
    ```

## Data Integration (ENTSO-E)
To fetch real Day-Ahead Prices for Greece:
1.  Get an API Key from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).
2.  Create a `.env` file in the root:
    ```
    ENTSOE_API_KEY=your_api_key_here
    ```
3.  Run the connector:
    ```bash
    python data/entsoe_connector.py
    ```

## Usage

### Training
To train the agent on the synthetic curriculum (Stage 1):
```bash
python agent/train.py --mode sine_wave
```


### Evaluation
To evaluate the trained agent against the Linear Programming benchmark:
```bash
python agent/evaluate.py
```
