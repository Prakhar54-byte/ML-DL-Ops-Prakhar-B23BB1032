# Assignment 2: Data Contracts & YAML
**Course:** ML-DL-Ops
**Name:** Prakhar Chauhan B23BB1032

## Overview
This submission contains four (4) distinct data contract YAML files. Each file addresses the specific data quality issues, logical mapping requirements, and enforcement strategies outlined in the assignment scenarios.

## Files Included

### 1. `rides_contract.yaml` (Scenario 1: Comprehensive)
* **Context:** Ride-share dynamic pricing model.
* **Key Features:**
  * **Logical Mapping:** Renamed cryptic DB columns (e.g., `ts_start` → `pickup_timestamp`, `r_id` → `ride_id`).
  * **Circuit Breaker:** Implemented `enforcement: hard` on the `fare_amount` rule. As requested, this stops the pipeline if a negative fare (e.g., -5.00) is detected to prevent the model crash.
  * **PII:** Explicitly tagged `passenger_id` as sensitive.
  * **SLA:** Defined a 30-minute freshness threshold for hourly model retraining.

### 2. `orders_contract.yaml` (Scenario 2: E-commerce)
* **Context:** Black Friday real-time dashboard.
* **Key Features:**
  * **Enum Mapping:** Mapped physical status codes (`2, 5, 9`) to logical values (`PAID, SHIPPED, CANCELLED`).
  * **Invalid Code Rejection:** Any status code not in the allowed list (like the bugged `7`) is automatically rejected by the schema validation.

### 3. `thermostat_contract.yaml` (Scenario 3: IoT)
* **Context:** Smart thermostat fleet monitoring.
* **Key Features:**
  * **Sanity Check:** Filters out the hardware error code `9999` by enforcing a strict temperature range (-30°C to 60°C).
  * **Battery Validation:** Ensures battery levels are within the valid `0.0` to `1.0` percentage range.

### 4. `fintech_contract.yaml` (Scenario 4: Financial)
* **Context:** Banking fraud detection system.
* **Key Features:**
  * **Regex Enforcement:** Used `^[A-Z0-9]{10}$` to validate Account IDs.
  * **Hard Stop:** Configured `enforcement: hard` to block the pipeline if 8-character legacy IDs are detected, preventing silent failures in the fraud model.

## Validation
All YAML files have been validated using `yamllint` to ensure correct syntax and indentation.