# Código de verificación inicial con NumPy y Matplotlib
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(7, 3.5))
plt.plot(x, y, color="#2563EB", linewidth=2, label="sin(x)")
plt.title("Gráfica elemental de verificación - Entorno ML")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()