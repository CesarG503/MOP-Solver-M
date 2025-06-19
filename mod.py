from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from fractions import Fraction

class GranMSimplexFractions:
    def __init__(self):
        self.M = "M"  # Usaremos M simbólico
        self.iteration = 0
        self.html_output = ""
        self.is_minimization = True
        
    def fraction_to_html(self, frac):
        """Convierte una fracción a HTML legible"""
        if isinstance(frac, str):
            return frac  # Para casos como "M"
        
        if isinstance(frac, (int, float)):
            frac = Fraction(frac).limit_denominator()
        
        if frac.denominator == 1:
            return str(frac.numerator)
        else:
            return f"{frac.numerator}/{frac.denominator}"
    
    def mixed_fraction_to_html(self, value):
        """Maneja fracciones mixtas con M"""
        if isinstance(value, str):
            return value
        
        if hasattr(value, 'coefficient') and hasattr(value, 'M_coefficient'):
            # Valor mixto: a + bM
            coef_str = self.fraction_to_html(value.coefficient) if value.coefficient != 0 else ""
            m_str = ""
            
            if value.M_coefficient != 0:
                if value.M_coefficient == 1:
                    m_str = "M"
                elif value.M_coefficient == -1:
                    m_str = "-M"
                else:
                    m_str = f"{self.fraction_to_html(value.M_coefficient)}M"
            
            if coef_str and m_str:
                if value.M_coefficient > 0:
                    return f"{coef_str} + {m_str}"
                else:
                    return f"{coef_str} {m_str}"
            elif coef_str:
                return coef_str
            elif m_str:
                return m_str
            else:
                return "0"
        
        return self.fraction_to_html(value)

class MixedValue:
    """Clase para manejar valores de la forma a + bM"""
    def __init__(self, coefficient=0, M_coefficient=0):
        self.coefficient = Fraction(coefficient).limit_denominator()
        self.M_coefficient = Fraction(M_coefficient).limit_denominator()
    
    def __add__(self, other):
        if isinstance(other, MixedValue):
            return MixedValue(
                self.coefficient + other.coefficient,
                self.M_coefficient + other.M_coefficient
            )
        else:
            return MixedValue(
                self.coefficient + Fraction(other).limit_denominator(),
                self.M_coefficient
            )
    
    def __sub__(self, other):
        if isinstance(other, MixedValue):
            return MixedValue(
                self.coefficient - other.coefficient,
                self.M_coefficient - other.M_coefficient
            )
        else:
            return MixedValue(
                self.coefficient - Fraction(other).limit_denominator(),
                self.M_coefficient
            )
    
    def __mul__(self, other):
        if isinstance(other, MixedValue):
            # (a + bM) * (c + dM) = ac + (ad + bc)M + bdM²
            # Asumimos que M² es muy grande, así que lo ignoramos en cálculos prácticos
            return MixedValue(
                self.coefficient * other.coefficient,
                self.coefficient * other.M_coefficient + self.M_coefficient * other.coefficient
            )
        else:
            other_frac = Fraction(other).limit_denominator()
            return MixedValue(
                self.coefficient * other_frac,
                self.M_coefficient * other_frac
            )
    
    def __truediv__(self, other):
        if isinstance(other, MixedValue):
            # División compleja, simplificamos asumiendo que other no tiene componente M
            if other.M_coefficient == 0:
                return MixedValue(
                    self.coefficient / other.coefficient,
                    self.M_coefficient / other.coefficient
                )
            else:
                raise ValueError("División por expresión con M no soportada")
        else:
            other_frac = Fraction(other).limit_denominator()
            return MixedValue(
                self.coefficient / other_frac,
                self.M_coefficient / other_frac
            )
    
    def __neg__(self):
        return MixedValue(-self.coefficient, -self.M_coefficient)
    
    def is_negative(self):
        """Determina si el valor es negativo (para criterio de optimalidad)"""
        # Si tiene componente M negativa, es muy negativo
        if self.M_coefficient < 0:
            return True
        elif self.M_coefficient > 0:
            return False
        else:
            return self.coefficient < 0
    
    def __str__(self):
        if self.coefficient == 0 and self.M_coefficient == 0:
            return "0"
        elif self.M_coefficient == 0:
            return str(self.coefficient)
        elif self.coefficient == 0:
            if self.M_coefficient == 1:
                return "M"
            elif self.M_coefficient == -1:
                return "-M"
            else:
                return f"{self.M_coefficient}M"
        else:
            m_part = ""
            if self.M_coefficient == 1:
                m_part = "M"
            elif self.M_coefficient == -1:
                m_part = "-M"
            elif self.M_coefficient != 0:
                m_part = f"{self.M_coefficient}M"
            
            if m_part:
                if self.M_coefficient > 0:
                    return f"{self.coefficient} + {m_part}"
                else:
                    return f"{self.coefficient} {m_part}"
            else:
                return str(self.coefficient)

class GranMSimplexFractions:
    def __init__(self):
        self.M = "M"
        self.iteration = 0
        self.html_output = ""
        self.is_minimization = True
        
    def fraction_to_html(self, frac):
        """Convierte una fracción a HTML legible"""
        if isinstance(frac, str):
            return frac
        
        if isinstance(frac, MixedValue):
            return self.mixed_value_to_html(frac)
        
        if isinstance(frac, (int, float)):
            frac = Fraction(frac).limit_denominator()
        
        if frac.denominator == 1:
            return str(frac.numerator)
        else:
            return f"<sup>{frac.numerator}</sup>&frasl;<sub>{frac.denominator}</sub>"
    
    def mixed_value_to_html(self, value, is_z_row=False):
        """Convierte MixedValue a HTML - MODIFICADO para cambiar signos en fila Z"""
        if value.coefficient == 0 and value.M_coefficient == 0:
            return "0"
        elif value.M_coefficient == 0:
            coef_to_show = -value.coefficient if is_z_row else value.coefficient
            return self.fraction_to_html(coef_to_show)
        elif value.coefficient == 0:
            m_coef_to_show = -value.M_coefficient if is_z_row else value.M_coefficient
            if m_coef_to_show == 1:
                return "M"
            elif m_coef_to_show == -1:
                return "-M"
            else:
                return f"{self.fraction_to_html(m_coef_to_show)}M"
        else:
            coef_to_show = -value.coefficient if is_z_row else value.coefficient
            m_coef_to_show = -value.M_coefficient if is_z_row else value.M_coefficient
            
            coef_str = self.fraction_to_html(coef_to_show)
            if m_coef_to_show == 1:
                m_str = "M"
            elif m_coef_to_show == -1:
                m_str = "-M"
            else:
                m_str = f"{self.fraction_to_html(m_coef_to_show)}M"
            
            if m_coef_to_show > 0:
                return f"{coef_str} + {m_str}"
            else:
                return f"{coef_str} {m_str}"
    
    def setup_problem(self, objective: List[float], constraints: List[List[float]], 
                     constraint_types: List[str], minimize: bool = True) -> Tuple[List[List], List[str], List[str]]:
        """Configura el problema inicial usando fracciones"""
        self.is_minimization = minimize
        num_vars = len(objective)
        num_constraints = len(constraints)
        
        # Contar variables auxiliares
        slack_needed = sum(1 for ct in constraint_types if ct in ['<=', '<'])
        surplus_needed = sum(1 for ct in constraint_types if ct in ['>=', '>'])
        artificial_needed = sum(1 for ct in constraint_types if ct in ['>=', '>', '='])
        
        total_aux_vars = slack_needed + surplus_needed + artificial_needed
        total_vars = num_vars + total_aux_vars
        
        # Nombres de variables
        original_vars = [f"x{i+1}" for i in range(num_vars)]
        slack_vars = []
        surplus_vars = []
        artificial_vars = []
        
        # Contadores
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        # Matriz extendida con MixedValue
        extended_matrix = []
        basis_vars = []
        
        # Procesar restricciones
        for i, (constraint, constraint_type) in enumerate(zip(constraints, constraint_types)):
            row = [MixedValue(0, 0) for _ in range(total_vars + 1)]
            
            # Variables originales
            for j in range(num_vars):
                row[j] = MixedValue(Fraction(constraint[j]).limit_denominator(), 0)
            
            # RHS
            row[-1] = MixedValue(Fraction(constraint[-1]).limit_denominator(), 0)
            
            if constraint_type in ['<=', '<']:
                slack_count += 1
                slack_var = f"s{slack_count}"
                slack_vars.append(slack_var)
                slack_position = num_vars + slack_count - 1
                row[slack_position] = MixedValue(1, 0)
                basis_vars.append(slack_var)
                
            elif constraint_type in ['>=', '>']:
                surplus_count += 1
                artificial_count += 1
                surplus_var = f"e{surplus_count}"
                artificial_var = f"a{artificial_count}"
                surplus_vars.append(surplus_var)
                artificial_vars.append(artificial_var)
                
                surplus_position = num_vars + slack_needed + surplus_count - 1
                artificial_position = num_vars + slack_needed + surplus_needed + artificial_count - 1
                
                row[surplus_position] = MixedValue(-1, 0)
                row[artificial_position] = MixedValue(1, 0)
                basis_vars.append(artificial_var)
                
            elif constraint_type == '=':
                artificial_count += 1
                artificial_var = f"a{artificial_count}"
                artificial_vars.append(artificial_var)
                
                artificial_position = num_vars + slack_needed + surplus_needed + artificial_count - 1
                row[artificial_position] = MixedValue(1, 0)
                basis_vars.append(artificial_var)
            
            extended_matrix.append(row)
        
        # Fila de función objetivo
        z_row = [MixedValue(0, 0) for _ in range(total_vars + 1)]
        
        # Coeficientes de variables originales
        for i, coef in enumerate(objective):
            coef_frac = Fraction(coef).limit_denominator()
            if minimize:
                z_row[i] = MixedValue(coef_frac, 0)
            else:
                z_row[i] = MixedValue(coef_frac, 0)
        
        # Penalización para variables artificiales
        artificial_start_idx = num_vars + slack_needed + surplus_needed
        for i in range(len(artificial_vars)):
            z_row[artificial_start_idx + i] = MixedValue(0, 1)  # +M
        
        extended_matrix.append(z_row)
        
        # Nombres de variables
        all_var_names = original_vars + slack_vars + surplus_vars + artificial_vars + ["RHS"]
        
        return extended_matrix, all_var_names, basis_vars
    
    def eliminate_artificial_from_z(self, matrix: List[List], basis_vars: List[str], all_var_names: List[str]) -> List[List]:
        """Elimina variables artificiales de la fila Z"""
        artificial_indices = [i for i, var in enumerate(all_var_names[:-1]) if var.startswith('a')]
        
        for art_idx in artificial_indices:
            if matrix[-1][art_idx].M_coefficient != 0 or matrix[-1][art_idx].coefficient != 0:
                # Encontrar fila donde está la variable artificial
                for i, basis_var in enumerate(basis_vars):
                    if basis_var == all_var_names[art_idx] and i < len(matrix) - 1:
                        multiplier = matrix[-1][art_idx]
                        for j in range(len(matrix[0])):
                            matrix[-1][j] = matrix[-1][j] - multiplier * matrix[i][j]
                        break
        
        return matrix
    
    def find_pivot(self, matrix: List[List]) -> Tuple[int, int]:
        """Encuentra el elemento pivote"""
        z_row = matrix[-1][:-1]
        
        # Encontrar el más negativo
        most_negative_idx = -1
        most_negative_val = None
        
        for i, val in enumerate(z_row):
            if val.is_negative():
                if most_negative_val is None or self.is_more_negative(val, most_negative_val):
                    most_negative_val = val
                    most_negative_idx = i
        
        if most_negative_idx == -1:
            return -1, -1  # Óptimo encontrado
        
        pivot_col = most_negative_idx
        
        # Prueba de la razón
        ratios = []
        for i in range(len(matrix) - 1):
            if matrix[i][pivot_col].coefficient > 0 or matrix[i][pivot_col].M_coefficient > 0:
                # Calcular razón RHS / elemento_columna
                rhs = matrix[i][-1]
                divisor = matrix[i][pivot_col]
                
                # Solo consideramos casos donde el divisor es positivo y sin M
                if divisor.M_coefficient == 0 and divisor.coefficient > 0:
                    if rhs.M_coefficient == 0:  # RHS sin M
                        ratio = rhs.coefficient / divisor.coefficient
                        ratios.append((ratio, i))
                    else:
                        ratios.append((float('inf'), i))
                else:
                    ratios.append((float('inf'), i))
            else:
                ratios.append((float('inf'), i))
        
        valid_ratios = [(r, i) for r, i in ratios if r != float('inf')]
        if not valid_ratios:
            return -1, -1  # No acotado
        
        min_ratio, pivot_row = min(valid_ratios)
        return pivot_row, pivot_col
    
    def is_more_negative(self, val1, val2):
        """Compara si val1 es más negativo que val2"""
        # Prioridad: M negativo > constante negativa
        if val1.M_coefficient < val2.M_coefficient:
            return True
        elif val1.M_coefficient > val2.M_coefficient:
            return False
        else:
            return val1.coefficient < val2.coefficient
    
    def pivot_operation(self, matrix: List[List], pivot_row: int, pivot_col: int, 
                       basis_vars: List[str], all_var_names: List[str]) -> List[List]:
        """Realiza operación de pivoteo con fracciones"""
        pivot_element = matrix[pivot_row][pivot_col]
        
        # HTML para pivoteo
        self.html_output += f"<p>Variable que entra: <strong>{all_var_names[pivot_col]}</strong> | "
        self.html_output += f"Variable que sale: <strong>{basis_vars[pivot_row]}</strong></p>"
        self.html_output += f"<p>Elemento pivote: <strong>{self.mixed_value_to_html(pivot_element)}</strong></p>"
        
        # Normalizar fila pivote
        self.html_output += f"<h3>Normalización de la fila pivote F{pivot_row+1}:</h3><ul>"
        for j in range(len(matrix[0])):
            old_val = matrix[pivot_row][j]
            matrix[pivot_row][j] = matrix[pivot_row][j] / pivot_element
            var_name = all_var_names[j] if j < len(all_var_names) else 'RHS'
            self.html_output += f"<li>{var_name}: {self.mixed_value_to_html(old_val)} ÷ {self.mixed_value_to_html(pivot_element)} = {self.mixed_value_to_html(matrix[pivot_row][j])}</li>"
        self.html_output += "</ul>"
        
        # Actualizar otras filas
        self.html_output += f"<h3>Actualización de las otras filas:</h3>"
        for i in range(len(matrix)):
            if i != pivot_row:
                factor = matrix[i][pivot_col]
                if factor.coefficient != 0 or factor.M_coefficient != 0:
                    self.html_output += f"<h4>F{i+1} = F{i+1} - ({self.mixed_value_to_html(factor)}) × F{pivot_row+1}</h4><ul>"
                    for j in range(len(matrix[0])):
                        old_val = matrix[i][j]
                        pivot_val = matrix[pivot_row][j]
                        matrix[i][j] = matrix[i][j] - factor * pivot_val
                        var_name = all_var_names[j] if j < len(all_var_names) else 'RHS'
                        # Mostrar con signo cambiado si es fila Z
                        is_z_row = (i == len(matrix) - 1)
                        self.html_output += f"<li>{var_name}: {self.mixed_value_to_html(old_val, is_z_row)} - ({self.mixed_value_to_html(factor)} × {self.mixed_value_to_html(pivot_val)}) = {self.mixed_value_to_html(matrix[i][j], is_z_row)}</li>"
                    self.html_output += "</ul>"
        
        # Actualizar base
        basis_vars[pivot_row] = all_var_names[pivot_col]
        
        return matrix
    
    def create_table_html(self, matrix: List[List], all_var_names: List[str], 
                         basis_vars: List[str], pivot_row: int = -1, pivot_col: int = -1) -> str:
        """Crea tabla HTML con fracciones - MODIFICADO para mostrar signos cambiados en fila Z"""
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
            
            # Determinar si es la fila Z
            is_z_row = (i == len(matrix) - 1)
            
            for j in range(len(matrix[0])):
                css_class = ""
                if i == pivot_row and j == pivot_col:
                    css_class = ' class="pivot"'
                elif i == pivot_row or j == pivot_col:
                    css_class = ' class="pivot-highlight"'
                
                html += f'<td{css_class}>{self.mixed_value_to_html(matrix[i][j], is_z_row)}</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        return html
    
    def solve(self, objective: List[float], constraints: List[List[float]], 
              constraint_types: List[str], minimize: bool = True) -> str:
        """Resuelve usando fracciones exactas"""
        
        # HTML inicial
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
                sup { font-size: 0.8em; }
                sub { font-size: 0.8em; }
            </style>
        </head>
        <body>
            <h1>Método de la Gran M - Cálculos Exactos con Fracciones (Signos Z Corregidos)</h1>
        """
        
        # Modelo original
        self.html_output += '<div class="model">'
        self.html_output += '<h2>Modelo Matemático Original</h2>'
        
        obj_str = "Minimizar " if minimize else "Maximizar "
        obj_str += "Z = "
        for i, coef in enumerate(objective):
            frac = Fraction(coef).limit_denominator()
            if i > 0 and frac >= 0:
                obj_str += " + "
            elif frac < 0:
                obj_str += " - " if i > 0 else "-"
                frac = abs(frac)
            
            if frac.denominator == 1:
                obj_str += f"{frac.numerator}x{i+1}"
            else:
                obj_str += f"({frac.numerator}/{frac.denominator})x{i+1}"
        
        self.html_output += f'<p><strong>{obj_str}</strong></p>'
        
        # Restricciones
        self.html_output += '<p><strong>Sujeto a:</strong></p><ul>'
        for i, (constraint, constraint_type) in enumerate(zip(constraints, constraint_types)):
            constraint_str = ""
            for j, coef in enumerate(constraint[:-1]):
                frac = Fraction(coef).limit_denominator()
                if j > 0 and frac >= 0:
                    constraint_str += " + "
                elif frac < 0:
                    constraint_str += " - " if j > 0 else "-"
                    frac = abs(frac)
                
                if frac.denominator == 1:
                    constraint_str += f"{frac.numerator}x{j+1}"
                else:
                    constraint_str += f"({frac.numerator}/{frac.denominator})x{j+1}"
            
            rhs_frac = Fraction(constraint[-1]).limit_denominator()
            if rhs_frac.denominator == 1:
                constraint_str += f" {constraint_type} {rhs_frac.numerator}"
            else:
                constraint_str += f" {constraint_type} {rhs_frac.numerator}/{rhs_frac.denominator}"
            
            self.html_output += f'<li>{constraint_str}</li>'
        
        var_list = ", ".join([f"x{i+1}" for i in range(len(objective))])
        self.html_output += f'<li>{var_list} ≥ 0</li>'
        self.html_output += '</ul></div>'
        
        # Configurar y resolver
        matrix, all_var_names, basis_vars = self.setup_problem(objective, constraints, constraint_types, minimize)
        
        self.html_output += '<h2>Proceso de Solución con Fracciones Exactas (Signos Z Visuales Corregidos)</h2>'
        
        # Eliminar artificiales de Z
        matrix = self.eliminate_artificial_from_z(matrix, basis_vars, all_var_names)
        
        # Iteraciones
        self.iteration = 0
        max_iterations = 50
        
        while self.iteration < max_iterations:
            pivot_row, pivot_col = self.find_pivot(matrix)
            self.html_output += self.create_table_html(matrix, all_var_names, basis_vars, pivot_row, pivot_col)
            
            if pivot_row == -1:
                self.html_output += '<p><strong>Solución óptima encontrada.</strong></p>'
                break
            
            matrix = self.pivot_operation(matrix, pivot_row, pivot_col, basis_vars, all_var_names)
            self.iteration += 1
        
        # Solución final
        self.html_output += '<h2>Solución Final</h2>'
        
        # Verificar factibilidad
        artificial_in_basis = False
        for i, var in enumerate(basis_vars):
            if var.startswith('a') and (matrix[i][-1].coefficient > 0 or matrix[i][-1].M_coefficient > 0):
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
                    solution[var] = matrix[row_idx][-1]
                else:
                    solution[var] = MixedValue(0, 0)
            
            # Valor objetivo - MOSTRAR CON SIGNO CAMBIADO
            z_value = matrix[-1][-1]
            z_value_display = MixedValue(-z_value.coefficient, -z_value.M_coefficient)
            if not minimize:
                z_value_display = MixedValue(-z_value_display.coefficient, -z_value_display.M_coefficient)
            
            self.html_output += f'<p><strong>Valor {"mínimo" if minimize else "máximo"} de la función objetivo: {self.mixed_value_to_html(z_value_display)}</strong></p>'
            self.html_output += '<p><strong>Valores de las variables:</strong></p><ul>'
            
            for i in range(len(objective)):
                var_name = f"x{i+1}"
                value = solution.get(var_name, MixedValue(0, 0))
                self.html_output += f'<li>{var_name} = {self.mixed_value_to_html(value)}</li>'
            
            self.html_output += '</ul>'
        
        self.html_output += '</body></html>'
        return self.html_output

# Función de prueba
def test_fractions():
    solver = GranMSimplexFractions()
    
    print("=== MÉTODO DE LA GRAN M CON FRACCIONES EXACTAS (SIGNOS Z CORREGIDOS) ===")
    
     # Ejemplo 1: Minimización
    print("\n1. PROBLEMA DE MINIMIZACIÓN:")
    print("Minimizar Z = 2x1 + 3x2")
    print("Sujeto a:")
    print("  x1 + x2 >= 4")
    print("  2x1 + x2 >= 6")
    print("  x1, x2 >= 0")
    
    objetivo = [2, 3]
    restricciones = [
        [1, 1, 4],   # x1 + x2 >= 4
        [2, 1, 6],   # 2x1 + x2 >= 6  
    ]
    tipos = ['>=', '>=']
    
    html_report = solver.solve(objetivo, restricciones, tipos, minimize=True)
    
    with open("gran_m_fractions_signos_corregidos.html", "w", encoding="utf-8") as f:
        f.write(html_report)
    
    print("✅ Reporte con signos Z corregidos generado: gran_m_fractions_signos_corregidos.html")
    
    return html_report

if __name__ == "__main__":
    test_fractions()
