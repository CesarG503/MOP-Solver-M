from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

class GranMSimplex:
    def __init__(self):
        self.M = 1000000  # Valor grande para el método de la Gran M
        self.iteration = 0
        self.html_output = ""
        self.is_minimization = True
        
    def setup_problem(self, objective: List[float], constraints: List[List[float]], 
                     constraint_types: List[str], minimize: bool = True) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Configura el problema inicial para el método de la Gran M
        """
        self.is_minimization = minimize
        num_vars = len(objective)
        num_constraints = len(constraints)
        
        # Primero, contar cuántas variables auxiliares necesitamos
        slack_needed = sum(1 for ct in constraint_types if ct in ['<=', '<'])
        surplus_needed = sum(1 for ct in constraint_types if ct in ['>=', '>'])
        artificial_needed = sum(1 for ct in constraint_types if ct in ['>=', '>', '='])
        
        total_aux_vars = slack_needed + surplus_needed + artificial_needed
        total_vars = num_vars + total_aux_vars
        
        # Listas para nombres de variables
        original_vars = [f"x{i+1}" for i in range(num_vars)]
        slack_vars = []
        surplus_vars = []
        artificial_vars = []
        
        # Contadores
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        # Matriz extendida con tamaño consistente
        extended_matrix = []
        basis_vars = []
        
        # Procesar cada restricción
        for i, (constraint, constraint_type) in enumerate(zip(constraints, constraint_types)):
            # Crear fila con el tamaño correcto (variables originales + auxiliares + RHS)
            row = [0.0] * (total_vars + 1)
            
            # Copiar coeficientes de variables originales
            for j in range(num_vars):
                row[j] = float(constraint[j])
            
            # RHS
            rhs = float(constraint[-1])
            row[-1] = rhs
            
            current_aux_idx = num_vars
            
            if constraint_type in ['<=', '<']:
                # Agregar variable de holgura
                slack_count += 1
                slack_var = f"s{slack_count}"
                slack_vars.append(slack_var)
                
                # Encontrar la posición correcta para esta variable de holgura
                slack_position = num_vars + slack_count - 1
                row[slack_position] = 1.0
                basis_vars.append(slack_var)
                
            elif constraint_type in ['>=', '>']:
                # Agregar variable de exceso y artificial
                surplus_count += 1
                artificial_count += 1
                surplus_var = f"e{surplus_count}"
                artificial_var = f"a{artificial_count}"
                surplus_vars.append(surplus_var)
                artificial_vars.append(artificial_var)
                
                # Posiciones para exceso y artificial
                surplus_position = num_vars + slack_needed + surplus_count - 1
                artificial_position = num_vars + slack_needed + surplus_needed + artificial_count - 1
                
                row[surplus_position] = -1.0  # Variable de exceso
                row[artificial_position] = 1.0  # Variable artificial
                basis_vars.append(artificial_var)
                
            elif constraint_type == '=':
                # Solo agregar variable artificial
                artificial_count += 1
                artificial_var = f"a{artificial_count}"
                artificial_vars.append(artificial_var)
                
                artificial_position = num_vars + slack_needed + surplus_needed + artificial_count - 1
                row[artificial_position] = 1.0
                basis_vars.append(artificial_var)
            
            extended_matrix.append(row)
        
        # Crear fila de la función objetivo
        z_row = [0.0] * (total_vars + 1)
        
        # Coeficientes de variables originales
        for i, coef in enumerate(objective):
            if minimize:
                z_row[i] = float(coef)  # Para minimización
            else:
                z_row[i] = -float(coef)  # Para maximización (convertir a minimización)
        
        # Penalización para variables artificiales
        artificial_start_idx = num_vars + slack_needed + surplus_needed
        for i in range(len(artificial_vars)):
            if minimize:
                z_row[artificial_start_idx + i] = self.M  # +M para minimización
            else:
                z_row[artificial_start_idx + i] = self.M  # +M (ya convertimos a minimización)
        
        extended_matrix.append(z_row)
        
        # Crear nombres de todas las variables
        all_var_names = original_vars + slack_vars + surplus_vars + artificial_vars + ["RHS"]
        
        return np.array(extended_matrix, dtype=float), all_var_names, basis_vars
    
    def eliminate_artificial_from_z(self, matrix: np.ndarray, basis_vars: List[str], all_var_names: List[str]) -> np.ndarray:
        """Elimina las variables artificiales de la fila Z"""
        artificial_indices = [i for i, var in enumerate(all_var_names[:-1]) if var.startswith('a')]
        
        for art_idx in artificial_indices:
            if abs(matrix[-1, art_idx]) > 1e-10:  # Si hay coeficiente no cero en Z
                # Encontrar la fila donde esta variable artificial está en la base
                for i, basis_var in enumerate(basis_vars):
                    if basis_var == all_var_names[art_idx] and i < len(matrix) - 1:
                        # Restar M veces esa fila de la fila Z
                        multiplier = matrix[-1, art_idx]
                        for j in range(len(matrix[0])):
                            matrix[-1, j] -= multiplier * matrix[i, j]
                        break
        
        return matrix
    
    def find_pivot(self, matrix: np.ndarray) -> Tuple[int, int]:
        """Encuentra el elemento pivote"""
        # Encontrar la columna pivote (más negativo en fila Z para minimización)
        z_row = matrix[-1, :-1]
        pivot_col = np.argmin(z_row)
        
        if z_row[pivot_col] >= -1e-10:  # Usar tolerancia numérica
            return -1, -1  # Solución óptima encontrada
        
        # Prueba de la razón para encontrar fila pivote
        ratios = []
        for i in range(len(matrix) - 1):
            if matrix[i, pivot_col] > 1e-10:  # Usar tolerancia numérica
                ratios.append(matrix[i, -1] / matrix[i, pivot_col])
            else:
                ratios.append(float('inf'))
        
        if all(r == float('inf') for r in ratios):
            return -1, -1  # Problema no acotado
        
        pivot_row = ratios.index(min(ratios))
        return pivot_row, pivot_col
    
    def pivot_operation(self, matrix: np.ndarray, pivot_row: int, pivot_col: int, 
                       basis_vars: List[str], all_var_names: List[str]) -> np.ndarray:
        """Realiza la operación de pivoteo"""
        pivot_element = matrix[pivot_row, pivot_col]
        
        # Detalles del pivoteo para el HTML
        self.html_output += f"<p>Variable que entra: <strong>{all_var_names[pivot_col]}</strong> | "
        self.html_output += f"Variable que sale: <strong>{basis_vars[pivot_row]}</strong></p>"
        self.html_output += f"<p>Elemento pivote: <strong>{pivot_element:.6f}</strong> en fila {pivot_row + 1}, columna {all_var_names[pivot_col]}</p>"
        
        # Normalizar fila pivote
        self.html_output += f"<h3>Normalización de la fila pivote F{pivot_row+1}:</h3><ul>"
        for j in range(len(matrix[0])):
            old_val = matrix[pivot_row, j]
            matrix[pivot_row, j] /= pivot_element
            var_name = all_var_names[j] if j < len(all_var_names) else 'RHS'
            self.html_output += f"<li>{var_name}: {old_val:.6f} / {pivot_element:.6f} = {matrix[pivot_row, j]:.6f}</li>"
        self.html_output += "</ul>"
        
        # Actualizar otras filas
        self.html_output += f"<h3>Actualización de las otras filas:</h3>"
        for i in range(len(matrix)):
            if i != pivot_row:
                factor = matrix[i, pivot_col]
                if abs(factor) > 1e-10:  # Usar tolerancia numérica
                    self.html_output += f"<h4>F{i+1} = F{i+1} - ({factor:.6f}) × F{pivot_row+1}</h4><ul>"
                    for j in range(len(matrix[0])):
                        old_val = matrix[i, j]
                        matrix[i, j] -= factor * matrix[pivot_row, j]
                        var_name = all_var_names[j] if j < len(all_var_names) else 'RHS'
                        self.html_output += f"<li>{var_name}: {old_val:.6f} - ({factor:.6f} × {matrix[pivot_row, j] + factor * matrix[pivot_row, j]:.6f}) = {matrix[i, j]:.6f}</li>"
                    self.html_output += "</ul>"
        
        # Actualizar variable en la base
        basis_vars[pivot_row] = all_var_names[pivot_col]
        
        return matrix
    
    def create_table_html(self, matrix: np.ndarray, all_var_names: List[str], 
                         basis_vars: List[str], pivot_row: int = -1, pivot_col: int = -1) -> str:
        """Crea la tabla HTML para una iteración"""
        html = f'<div class="iteration-title">Iteración {self.iteration}</div>'
        html += '<table><thead><tr><th>Base</th>'
        
        for var_name in all_var_names:
            html += f'<th>{var_name}</th>'
        html += '</tr></thead><tbody>'
        
        for i in range(len(matrix)):
            html += '<tr>'
            if i < len(basis_vars):
                html += f'<td>{basis_vars[i]}</td>'
            else:
                html += '<td>Z</td>'
            
            for j in range(len(matrix[0])):
                css_class = ""
                if i == pivot_row and j == pivot_col:
                    css_class = ' class="pivot"'
                elif i == pivot_row or j == pivot_col:
                    css_class = ' class="pivot-highlight"'
                
                html += f'<td{css_class}>{matrix[i, j]:.6f}</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        return html
    
    def solve(self, objective: List[float], constraints: List[List[float]], 
              constraint_types: List[str], minimize: bool = True) -> str:
        """Resuelve el problema usando el método de la Gran M"""
        
        # Inicializar HTML
        self.html_output = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #003366; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #999; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; }
                .pivot { background-color: #ffcccc; font-weight: bold; }
                .pivot-highlight { background-color: #ffffcc; }
                .iteration-title { background-color: #003366; color: white; padding: 8px; margin-top: 20px; }
                .model { background-color: #f9f9f9; padding: 15px; border-left: 4px solid #003366; }
                ul { text-align: left; }
            </style>
        </head>
        <body>
            <h1>Método de la Gran M - Informe Completo</h1>
        """
        
        # Mostrar modelo matemático
        self.html_output += '<div class="model">'
        self.html_output += '<h2>Modelo Matemático Original</h2>'
        
        # Función objetivo
        obj_str = "Minimizar " if minimize else "Maximizar "
        obj_str += "Z = "
        for i, coef in enumerate(objective):
            if i > 0 and coef >= 0:
                obj_str += " + "
            elif coef < 0:
                obj_str += " - " if i > 0 else "-"
                coef = abs(coef)
            else:
                obj_str += ""
            obj_str += f"{coef}x{i+1}"
        
        self.html_output += f'<p><strong>{obj_str}</strong></p>'
        
        # Restricciones
        self.html_output += '<p><strong>Sujeto a:</strong></p><ul>'
        for i, (constraint, constraint_type) in enumerate(zip(constraints, constraint_types)):
            constraint_str = ""
            for j, coef in enumerate(constraint[:-1]):
                if j > 0 and coef >= 0:
                    constraint_str += " + "
                elif coef < 0:
                    constraint_str += " - " if j > 0 else "-"
                    coef = abs(coef)
                constraint_str += f"{coef}x{j+1}"
            constraint_str += f" {constraint_type} {constraint[-1]}"
            self.html_output += f'<li>{constraint_str}</li>'
        
        # Variables no negativas
        var_list = ", ".join([f"x{i+1}" for i in range(len(objective))])
        self.html_output += f'<li>{var_list} ≥ 0</li>'
        self.html_output += '</ul></div>'
        
        # Configurar problema
        matrix, all_var_names, basis_vars = self.setup_problem(objective, constraints, constraint_types, minimize)
        
        # Mostrar transformación
        self.html_output += '<h2>Transformación del Problema</h2>'
        self.html_output += '<p>Después de agregar variables de holgura, exceso y artificiales:</p>'
        
        # Eliminar variables artificiales de la fila Z
        matrix = self.eliminate_artificial_from_z(matrix, basis_vars, all_var_names)
        
        # Iteraciones del método simplex
        self.iteration = 0
        max_iterations = 50
        
        while self.iteration < max_iterations:
            # Mostrar tabla actual
            pivot_row, pivot_col = self.find_pivot(matrix)
            self.html_output += self.create_table_html(matrix, all_var_names, basis_vars, pivot_row, pivot_col)
            
            if pivot_row == -1:
                self.html_output += '<p><strong>Solución óptima encontrada.</strong></p>'
                break
            
            # Realizar pivoteo
            matrix = self.pivot_operation(matrix, pivot_row, pivot_col, basis_vars, all_var_names)
            self.iteration += 1
        
        # Mostrar solución final
        self.html_output += '<h2>Solución Final</h2>'
        
        # Verificar si hay variables artificiales en la base con valor > 0
        artificial_in_basis = False
        for i, var in enumerate(basis_vars):
            if var.startswith('a') and matrix[i, -1] > 1e-6:
                artificial_in_basis = True
                break
        
        if artificial_in_basis:
            self.html_output += '<p><strong>El problema no tiene solución factible.</strong></p>'
        else:
            # Extraer solución
            solution = {}
            for i, var in enumerate(all_var_names[:-1]):
                if var in basis_vars:
                    row_idx = basis_vars.index(var)
                    solution[var] = matrix[row_idx, -1]
                else:
                    solution[var] = 0
            
            # Valor de la función objetivo
            z_value = matrix[-1, -1]
            if not minimize:
                z_value = -z_value  # Convertir de vuelta para maximización
            
            self.html_output += f'<p><strong>Valor {"mínimo" if minimize else "máximo"} de la función objetivo: {z_value:.6f}</strong></p>'
            self.html_output += '<p><strong>Valores de las variables:</strong></p><ul>'
            
            for i in range(len(objective)):
                var_name = f"x{i+1}"
                value = solution.get(var_name, 0)
                self.html_output += f'<li>{var_name} = {value:.6f}</li>'
            
            self.html_output += '</ul>'
        
        self.html_output += '</body></html>'
        return self.html_output

# Ejemplo de uso y prueba
def main():
    # Crear instancia del solver
    solver = GranMSimplex()
    
    # Ejemplo del usuario: Maximizar Z = 4X1 + 3X2 + 2X3
    # Sujeto a: X1 + X2 + X3 ≤ 10, 2X1 + X2 ≤ 8, X1, X2, X3 ≥ 0
    objetivo = [4, 3, 2]
    restricciones = [
        [1, 1, 1, 10],  # X1 + X2 + X3 ≤ 10
        [2, 1, 0, 8],   # 2X1 + X2 ≤ 8
    ]
    tipos_restricciones = ['<=', '<=']
    
    print("Resolviendo problema de programación lineal...")
    print("Maximizar Z = 4X1 + 3X2 + 2X3")
    print("Sujeto a:")
    print("  X1 + X2 + X3 ≤ 10")
    print("  2X1 + X2 ≤ 8")
    print("  X1, X2, X3 ≥ 0")
    print()
    
    # Resolver (False = maximización)
    html_report = solver.solve(objetivo, restricciones, tipos_restricciones, minimize=False)
    
    # Guardar reporte
    with open("gran_m_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)
    
    print("✅ Reporte generado exitosamente: gran_m_report.html")
    print("El archivo HTML contiene el análisis completo paso a paso del método de la Gran M.")
    
    return html_report

if __name__ == "__main__":
    result = main()
