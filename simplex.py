from typing import List, Tuple
import pandas as pd

def prepare_initial_table(obj: List[float], constraints: List[List[float]]) -> Tuple[List[List[float]], List[str]]:
    num_vars = len(obj)
    num_constraints = len(constraints)

    matrix = []
    for i, row in enumerate(constraints):
        vars_part = row[:-1]
        rhs = row[-1]
        slack = [0] * num_constraints
        slack[i] = 1
        matrix.append(vars_part + slack + [rhs])

    z_row = [-c for c in obj] + [0] * (num_constraints + 1)
    matrix.append(z_row)

    var_names = [f"x{i+1}" for i in range(num_vars)]
    return matrix, var_names

def generate_html_simplex_report(matrix: List[List[float]], var_names: List[str]) -> str:
    num_constraints = len(matrix) - 1
    num_vars = len(var_names)

    slack_vars = [f"s{i+1}" for i in range(num_constraints)]
    all_vars = var_names + slack_vars + ["RHS"]
    table = pd.DataFrame(matrix, columns=all_vars)

    basis = slack_vars.copy()

    html_output = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h2 { color: #003366; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #999; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            .pivot { background-color: #ffcccc; font-weight: bold; }
            .iteration-title { background-color: #003366; color: white; padding: 8px; }
        </style>
    </head>
    <body>
        <h1>Método Simplex - Informe Paso a Paso</h1>
    """

    iteration = 0
    while True:
        html_output += f'<div class="iteration-title">Iteración {iteration}</div>'
        html_output += "<table><thead><tr><th>Base</th>"
        for col in table.columns:
            html_output += f"<th>{col}</th>"
        html_output += "</tr></thead><tbody>"

        for i in range(len(table)):
            html_output += f"<tr><td>{basis[i] if i < len(basis) else 'Z'}</td>"
            for j, col in enumerate(table.columns):
                val = table.iloc[i, j]
                html_output += f"<td>{val:.6f}</td>"
            html_output += "</tr>"
        html_output += "</tbody></table>"

        last_row = table.iloc[-1, :-1]
        pivot_col_name = last_row.idxmin()
        if table.at[len(table) - 1, pivot_col_name] >= 0:
            html_output += "<p><strong>Solución óptima encontrada.</strong></p>"
            break

        # Ratio test
        ratios = []
        for i in range(len(table) - 1):
            col_val = table.at[i, pivot_col_name]
            if col_val > 0:
                ratios.append(table.at[i, "RHS"] / col_val)
            else:
                ratios.append(float('inf'))

        pivot_row = ratios.index(min(ratios))
        pivot_element = table.at[pivot_row, pivot_col_name]

        entering = pivot_col_name
        leaving = basis[pivot_row]

        html_output += f"<p>Variable que entra: <strong>{entering}</strong> | Variable que sale: <strong>{leaving}</strong></p>"
        html_output += f"<p>Elemento pivote: <strong>{pivot_element:.6f}</strong> en fila {pivot_row + 1}, columna {pivot_col_name}</p>"

        old_pivot_row = table.iloc[pivot_row].copy()

        # Normalizar fila pivote con detalles
        html_output += f"<h3>Normalización de la fila pivote F{pivot_row+1}:</h3><ul>"
        new_pivot_row = []
        for col in table.columns:
            original_val = table.at[pivot_row, col]
            new_val = original_val / pivot_element
            new_pivot_row.append(new_val)
            html_output += f"<li>{col}: {original_val:.6f} / {pivot_element:.6f} = {new_val:.6f}</li>"
        table.iloc[pivot_row] = new_pivot_row
        html_output += "</ul>"

        # Actualización de las otras filas
        html_output += f"<h3>Actualización de las otras filas:</h3>"
        for i in range(len(table)):
            if i != pivot_row:
                factor = table.at[i, pivot_col_name]
                html_output += f"<h4>F{i+1} = F{i+1} - ({factor:.6f}) × F{pivot_row+1}</h4><ul>"
                original_row = table.iloc[i].copy()
                new_row = []
                for col in table.columns:
                    old_val = original_row[col]
                    pivot_val = table.at[pivot_row, col]
                    result = old_val - factor * pivot_val
                    new_row.append(result)
                    html_output += f"<li>{col}: {old_val:.6f} - ({factor:.6f} × {pivot_val:.6f}) = {result:.6f}</li>"
                table.iloc[i] = new_row
                html_output += "</ul>"

        basis[pivot_row] = entering
        iteration += 1

    html_output += "</body></html>"
    return html_output

# Ejemplo
objetivo = [3, 5]
restricciones = [
    [2, 3, 8],
    [4, 1, 6],
]

# Datos de entrada
objetivo = [4, 3]
restricciones = [
    [2, 1, 10],
    [1, 2, 12],
    [1, 1, 8],
]

matrix, var_names = prepare_initial_table(objetivo, restricciones)
html_report = generate_html_simplex_report(matrix, var_names)

with open("simplex_report_ejemplo.html", "w", encoding="utf-8") as f:
    f.write(html_report)

print("Reporte generado: simplex_report_ejemplo.html")
