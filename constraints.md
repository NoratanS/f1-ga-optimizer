# F1 Calendar Optimization - Operations Research Model

## 1. Sets and Indices
* $V = \{1, \dots, 24\}$ : Set of Formula 1 Tracks (Races).
* $W = \{1, \dots, 40\}$ : Set of Season Weeks (Slots).
* $i, j \in V$ : Track indices.
* $t \in W$ : Week indices.
* $k \in \{1, \dots, 24\}$ : Chronological index of the race in the season.

## 2. Decision Variables
### 2.1. Binary Assignment Variables
* $x_{i,t} \in \{0, 1\}$ : 1 if race $i$ is held in week $t$, 0 otherwise.
* $y_t \in \{0, 1\}$ : 1 if any race is held in week $t$ (race week).
  $$y_t = \sum_{i \in V} x_{i,t}$$

### 2.2. Sequence Variables (Routing)
Let the valid schedule be represented as an ordered sequence of races:
* $\omega_k \in V$ : The track assigned to the $k$-th race of the season.
* $\tau_k \in W$ : The week number in which the $k$-th race takes place.

## 3. Parameters
### 3.1. Input Data
* $D_{i,j}$ : Great-circle distance between track $i$ and $j$ (km).
* $T_{i,t}$ : Expected temperature at track $i$ during week $t$.
* $UTC_i$ : Timezone offset of track $i$.
* $R_i$ : Categorical geographic region of track $i$ (e.g., Europe, Middle East).
* $S = [t_{start}, t_{end}]$ : Summer break window (e.g., weeks 18 to 26).
* $W_{ram}$ : Set of weeks overlapping with Ramadan (e.g., weeks 7 to 11).
* $V_{ME}$ : Subset of tracks located in the Middle East ($V_{ME} \subset V$).

### 3.2. Constraints & Thresholds
* $[T^{min}_{hard}, T^{max}_{hard}]$ : Critical weather limits (5°C, 40°C).
* $[T^{min}_{opt}, T^{max}_{opt}]$ : Optimal weather window (18°C, 28°C).

### 3.3. Base Penalty Factors (Scaling Constants)
* $K_{env}$ : Penalty per 1°C deviation (e.g., 1000 pts/°C).
* $K_{utc}$ : Base penalty for timezone difference (e.g., 500 pts).
* $K_{dist}$ : Base penalty for distance (e.g., 1 pt/km).
* $K_{double}$ : Flat penalty for a double-header (e.g., 5000 pts).
* $K_{triple}$ : Flat penalty for a triple-header (e.g., 15000 pts).
* $K_{bounce}$ : Flat penalty for changing regions (e.g., 10000 pts).
* $K_{flyaway}$ : Heavy penalty for inter-continental back-to-backs (e.g., 50000 pts).
* $M$ : Big-M constant for Hard Constraint Violations (e.g., 1,000,000 pts).

### 3.4. Objective Weights (Sum = 1.0)
* $w_{dist}$ , $w_{env}$ , $w_{log}$ , $w_{hum}$

---

## 4. Hard Constraints (HC) - Violation Counting Functions
For the Genetic Algorithm to evaluate invalid schedules, hard constraints are formulated as functions that count the number of violations. Let $\mathbb{I}$ represent an indicator function returning 1 if the condition is true, 0 otherwise.

* **[HC1] Race Completeness ($V_{comp}$):**
  Penalizes any race scheduled zero times or more than once.
  $$V_{comp} = \sum_{i=1}^{24} \left| 1 - \sum_{t=1}^{40} x_{i,t} \right|$$

* **[HC2] Capacity Constraint ($V_{cap}$):**
  Counts instances where more than one race is scheduled in a single week.
  $$V_{cap} = \sum_{t=1}^{40} \max\left(0, \left(\sum_{i=1}^{24} x_{i,t}\right) - 1\right)$$

* **[HC3] Extreme Weather ($V_{weather}$):**
  Counts scheduled races falling outside the hard temperature limits.
  $$V_{weather} = \sum_{t=1}^{40} \sum_{i=1}^{24} x_{i,t} \cdot \mathbb{I}(T_{i,t} < T^{min}_{hard} \text{ or } T_{i,t} > T^{max}_{hard})$$

* **[HC4] Burnout Prevention ($V_{burnout}$):**
  Counts instances of 4 or more consecutive race weeks.
  $$V_{burnout} = \sum_{t=1}^{37} \max(0, y_t + y_{t+1} + y_{t+2} + y_{t+3} - 3)$$

* **[HC5] Summer Break Requirement ($V_{break}$):**
  Returns 1 if there is no consecutive 3-week gap within the summer window $S$, 0 otherwise.
  $$V_{break} = 1 - \max_{t \in S} \Big( \mathbb{I}(y_t + y_{t+1} + y_{t+2} = 0) \Big)$$

* **[HC6] Ramadan Observance ($V_{ramadan}$):**
  Counts Middle Eastern races placed during the Ramadan period.
  $$V_{ramadan} = \sum_{i \in V_{ME}} \sum_{t \in W_{ram}} x_{i,t}$$

**Total Hard Constraint Violations ($V_{total}$):**
$$V_{total} = V_{comp} + V_{cap} + V_{weather} + V_{burnout} + V_{break} + V_{ramadan}$$

---

## 5. Soft Constraints (SC) - Penalties to Minimize

### 5.1. Distance Penalty ($Z_{dist}$)
Total physical distance traveled over the season, summed over all 23 consecutive race-to-race legs:
$$Z_{dist} = K_{dist} \cdot \sum_{k=1}^{23} D_{\omega_k, \omega_{k+1}}$$

The pairwise distance $D_{i,j}$ between two tracks is the **great-circle distance in kilometers**, computed via the **Haversine formula** from each track's latitude and longitude. This approximates the Earth as a sphere and gives the shortest surface distance between two points — a reasonable proxy for freight/travel cost without modeling specific air or sea routes.

Let $\varphi_i, \lambda_i$ denote the latitude and longitude (in radians) of track $i$, and let $R = 6371$ km be the mean radius of the Earth. Define the angular differences:
$$\Delta\varphi = \varphi_j - \varphi_i \qquad \Delta\lambda = \lambda_j - \lambda_i$$

The Haversine distance is then:
$$a = \sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos(\varphi_i) \cdot \cos(\varphi_j) \cdot \sin^2\!\left(\frac{\Delta\lambda}{2}\right)$$
$$D_{i,j} = 2R \cdot \arcsin\!\left(\sqrt{a}\right)$$

In the implementation, $D_{i,j}$ is precomputed once into a symmetric distance matrix at startup, so fitness evaluation during the GA loop is a constant-time lookup per leg.

### 5.2. Environmental Penalty ($Z_{env}$)
Linear penalty for every degree outside the optimal window:
$$Z_{env} = K_{env} \cdot \sum_{t=1}^{40} \sum_{i=1}^{24} x_{i,t} \cdot \max(0, T^{min}_{opt} - T_{i,t}, T_{i,t} - T^{max}_{opt})$$

### 5.3. Human Fatigue: Jet Lag ($Z_{utc}$)
Non-linear (quadratic) penalty for timezone jumps between consecutive races, accounting for the modulo 24 circular nature of global timezones.
Let $\Delta UTC_{k}$ be the shortest biological time shift:
$$\Delta UTC_{k} = \min(|UTC_{\omega_{k+1}} - UTC_{\omega_k}|, 24 - |UTC_{\omega_{k+1}} - UTC_{\omega_k}|)$$
$$Z_{utc} = K_{utc} \cdot \sum_{k=1}^{23} (\Delta UTC_k)^2$$

### 5.4. Human Fatigue: Schedule Intensity ($Z_{headers}$)
Mathematical counting of double and triple headers using the boolean week variable $y_t$. 
*(Note: A sequence of 3 races triggers the double-header penalty twice and the triple-header penalty once, which naturally compounds the fatigue cost).*
$$N_{double} = \sum_{t=1}^{39} (y_t \cdot y_{t+1})$$
$$N_{triple} = \sum_{t=1}^{38} (y_t \cdot y_{t+1} \cdot y_{t+2})$$
$$Z_{headers} = (K_{double} \cdot N_{double}) + (K_{triple} \cdot N_{triple})$$

### 5.5. Freight Logistics & Regional Grouping ($Z_{log}$)
Heavy penalties applied for inefficient sea-freight routing and extreme inter-continental air-freight demands. $\mathbb{I}$ represents an indicator function returning 1 if true, 0 otherwise.

**Region Bounce (Zig-Zag Tax):**
$$P_{bounce} = \sum_{k=1}^{23} \mathbb{I}(R_{\omega_k} \neq R_{\omega_{k+1}})$$

**Flyaway Back-to-Back Penalty:**
$$P_{flyaway} = \sum_{k=1}^{23} \Big[ \mathbb{I}(\tau_{k+1} - \tau_k = 1) \cdot \mathbb{I}(R_{\omega_k} \neq R_{\omega_{k+1}}) \Big]$$

**Total Logistics Penalty:**
$$Z_{log} = (K_{bounce} \cdot P_{bounce}) + (K_{flyaway} \cdot P_{flyaway})$$

---

## 6. Genetic Algorithm Fitness Function
To evolve the population, the Genetic Algorithm evaluates chromosomes based on a Total Fitness score ($F$), which applies the Big-M death penalty multiplier to the violation count alongside the weighted soft penalties. 

**Soft Target Objective ($Z$):**
$$Z = w_{dist} \cdot Z_{dist} + w_{env} \cdot Z_{env} + w_{log} \cdot Z_{log} + w_{hum} \cdot (Z_{utc} + Z_{headers})$$

**Total Fitness Score ($F$) to Minimize:**
$$F = Z + (M \cdot V_{total})$$
