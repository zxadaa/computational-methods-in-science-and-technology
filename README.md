# Computational Methods in Science and Technology

This repository contains programming assignments and numerical experiments completed for the university course **Computational Methods in Science and Technology**. The labs explore numerical algorithms, their accuracy and stability, and ways to analyze and visualize their results.

## Labs

- **Lab 1:** Floating-point precision and numerical stability.
- **Lab 2a:** Polynomial interpolation using Lagrange and Newton forms.
- **Lab 2b:** Hermite interpolation.
- **Lab 3:** Spline interpolation.
- **Lab 4a:** Algebraic polynomial interpolation.
- **Lab 4b:** Trigonometric interpolation.
- **Lab 5:** Nonlinear equation solving with Newton's method and the secant method.
- **Lab 6:** Linear systems, Gaussian elimination, conditioning, accuracy, runtime, and memory use.
- **Lab 7:** The Jacobi iterative method, convergence behavior, and spectral radius.
- **Lab 8:** Numerical solution of an ordinary differential equation using Euler's method and fourth-order Runge-Kutta (RK4).

## Repository Layout

Each lab has its own directory containing its source files and, where available, an `instruction.txt` file and a PDF report named `report...pdf`. Generated or collected plots are stored in `plots/`, and tabular data is stored in `results/`. Lab 4 is split into the `a/` and `b/` subdirectories.

```text
lab1/                 Floating-point precision
lab2a/                Polynomial interpolation
lab2b/                Hermite interpolation
lab3/                 Splines
lab4/a/               Algebraic interpolation
lab4/b/               Trigonometric interpolation
lab5/                 Root-finding methods
lab6/                 Linear systems
lab7/                 Jacobi method
lab8/                 Ordinary differential equations
```

## Running the Programs

Run each script from its lab directory.

The required Python version and packages may vary by lab. Check the lab's `instruction.txt` before running its scripts. Python dependencies used across the repository include NumPy, Matplotlib, Pandas, and Seaborn; Lab 1 may also require a C++ compiler, as described in its instructions.
