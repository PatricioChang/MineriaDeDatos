import pypdf
import pandas as pd
import re
import os
import glob

# 1. DICCIONARIO DE NORMALIZACIÓN (Raíces de palabras clave)
MAPEO_CATEGORIAS = {
    'Lenguaje': ['lengua', 'literat', 'comunicaci', 'lectura'],
    'Matematica': ['matemát', 'matemat', 'álgebra', 'cálculo', 'probabilidad', 'estadístic', 'pensamiento matem'],
    'Historia': ['historia', 'geograf', 'sociales', 'ciudadanía', 'chile', 'sociedad', 'formación ciudadana'],
    'Ciencias': ['naturales', 'biolog', 'físic', 'químic', 'salud', 'ciencia para la ciudadania'],
    'Ingles': ['inglés', 'ingles', 'extranjero'],
    'EdFisica': ['física y salud', 'ejercicio', 'deportivo', 'deportes', 'psicomotricidad'],
    'Artes': ['visuales', 'dibujo', 'pintura', 'escultura', 'danza', 'teatral', 'artística'],
    'Musica': ['música', 'musical'],
    'Tecnologia': ['tecnolog', 'informát', 'programaci', 'computacional'],
    'TP_Salud': ['medicament', 'primeros auxilios', 'bioseguridad', 'higiene', 'adulto mayor', 'enfermedad'],
    'TP_Admin': ['comercial', 'tributaria', 'logística', 'almacen', 'bodega', 'cliente', 'administrativos'],
    'TP_Geologia': ['geolog', 'rocas', 'minerales', 'sondaje', 'mapeo', 'topográfic'],
    'TP_Parvulos': ['párvulo', 'didáctico', 'ambientación', 'recreación', 'infancia', 'alimentación'],
    'ACLE': ['a.c.l.e.']
}

def limpiar_rut(val):
    if pd.isna(val): return ""
    return re.sub(r'[\.\-\s]', '', str(val)).upper()

def procesar_colegio_final(carpeta_pdfs):
    archivos = glob.glob(os.path.join(carpeta_pdfs, "*.pdf"))
    print(f"🚀 Iniciando Minería de Datos. Detectados: {len(archivos)} cursos.")
    
    # COORDENADAS AJUSTADAS (Calibración Quirúrgica)
    # P1 Final está en ~501. La nota 1 de P2 está en ~530. 
    # Usamos TOLERANCIA 10 para que no se mezclen.
    X_COL = {'P1': 501, 'P2': 861, 'Final': 956}
    TOLERANCIA = 10 
    
    dataset_final = []
    total_alumnos_global = 0

    for idx, ruta in enumerate(archivos):
        nombre_f = os.path.basename(ruta)
        print(f"📂 [{idx+1}/{len(archivos)}] Procesando: {nombre_f}...", end="")
        
        try:
            reader = pypdf.PdfReader(ruta)
            alumnos_en_este_pdf = 0
            
            for page in reader.pages:
                words = []
                # Captura de texto con coordenadas exactas
                page.extract_text(visitor_text=lambda t,cm,tm,fd,fs: words.append({'text':t.strip(), 'x':tm[4], 'y':tm[5]}) if t.strip() else None)
                
                lines = {}
                for w in words:
                    y = round(w['y'], 1)
                    if y not in lines: lines[y] = []
                    lines[y].append(w)

                student = {'Archivo_Origen': nombre_f}
                txt_full = page.extract_text().lower()
                
                # Identificación
                rut_match = re.search(r'rut:\s*\n?\s*([\d\.\-kks\s]+)', txt_full)
                if rut_match: student['RUT'] = limpiar_rut(rut_match.group(1))
                
                curso_match = re.search(r'curso:\s*\n?\s*([^\n]+)', txt_full)
                if curso_match: student['Curso_Oficial'] = curso_match.group(1).strip().upper()

                # Asistencia
                asist_list = re.findall(r'(\d+)%', txt_full)
                if len(asist_list) >= 2:
                    student['Asist_P1'], student['Asist_P2'] = float(asist_list[0]), float(asist_list[1])
                
                # Promedios Globales
                p_gen = re.findall(r'(\d[\.,]\d)', txt_full[txt_full.find("promedio general"):])
                if len(p_gen) >= 2:
                    student['Global_P1'] = float(p_gen[0].replace(',','.'))
                    student['Global_P2'] = float(p_gen[1].replace(',','.'))

                # Notas por Categoría
                for y in sorted(lines.keys(), reverse=True):
                    row = sorted(lines[y], key=lambda w: w['x'])
                    line_txt = " ".join([w['text'] for w in row]).lower()
                    
                    categoria = None
                    if 'educación física' in line_txt or 'ciencias del ejercicio' in line_txt:
                        categoria = 'EdFisica'
                    else:
                        for cat, keys in MAPEO_CATEGORIAS.items():
                            if any(k in line_txt for k in keys):
                                categoria = cat
                                break
                    
                    if categoria:
                        for w in row:
                            if re.match(r'^(\d[\.,]\d)$', w['text']):
                                val = float(w['text'].replace(',', '.'))
                                # El secreto está en esta validación de coordenadas:
                                if abs(w['x'] - X_COL['P1']) < TOLERANCIA: student[f'{categoria}_P1'] = val
                                elif abs(w['x'] - X_COL['P2']) < TOLERANCIA: student[f'{categoria}_P2'] = val
                                elif abs(w['x'] - X_COL['Final']) < TOLERANCIA: student[f'{categoria}_Final'] = val

                if 'RUT' in student:
                    dataset_final.append(student)
                    alumnos_en_este_pdf += 1
            
            total_alumnos_global += alumnos_en_este_pdf
            print(f" OK ({alumnos_en_este_pdf} alumnos)")
            
        except Exception as e:
            print(f" ERROR: {e}")

    # Exportación
    if dataset_final:
        df_master = pd.DataFrame(dataset_final)
        df_master.to_csv('dataset_maestro_corregido.csv', index=False)
        print(f"\n✨ ¡FINALIZADO! {total_alumnos_global} alumnos guardados en 'dataset_maestro_corregido.csv'.")
        return df_master

# Ejecutar:
df = procesar_colegio_final(r'f:/Proyectos/Mineria de Datos/mis_pdfs/')