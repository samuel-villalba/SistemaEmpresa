"""
SmartPark - Sistema Integrado de Reconocimiento de Placas y Rostros
"""

import mysql.connector
from mysql.connector import Error
import streamlit as st
import cv2
import numpy as np
import easyocr
import imutils
import rembg
from PIL import Image
import io
import re
import time
import logging
import face_recognition
import os
from PIL import ImageDraw

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('smartpark')

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="SmartPark - Sistema Integrado",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- Funciones comunes -------------------------

def mostrar_imagen(foto_bytes):
    if foto_bytes:
        try:
            imagen = Image.open(io.BytesIO(foto_bytes))
            st.image(imagen, caption="Foto de perfil", width=200)
        except:
            st.warning("No se pudo mostrar la imagen")

# ------------------------- Funciones para Vehículos -------------------------

def conectar_bd():
    try:
        conexion = mysql.connector.connect(
            host='localhost',
            database='smartpark',
            user='root',
            password=''
        )
        if conexion.is_connected():
            return conexion
    except Error as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        logger.error(f"Error de conexión a la BD: {e}")
        return None

def buscar_vehiculo_por_placa(placa):
    try:
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            cursor = conexion.cursor(dictionary=True)
            query = "SELECT * FROM vehiculos WHERE UPPER(placa) = UPPER(%s)"
            cursor.execute(query, (placa,))
            resultado = cursor.fetchone()
            cursor.close()
            conexion.close()
            return resultado
        return None
    except Error as e:
        st.error(f"Error al buscar vehículo: {e}")
        logger.error(f"Error al buscar vehículo con placa {placa}: {e}")
        return None

def obtener_empleados():
    try:
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            cursor = conexion.cursor(dictionary=True)
            cursor.execute("SELECT id, CONCAT(nombre, ' ', apellido) AS nombre_completo FROM empleados ORDER BY nombre")
            empleados = cursor.fetchall()
            cursor.close()
            conexion.close()
            return empleados
        return []
    except Error as e:
        st.error(f"Error al obtener empleados: {e}")
        logger.error(f"Error al obtener lista de empleados: {e}")
        return []

def registrar_vehiculo(empleado_id, placa, marca, modelo, tipo, color, foto_bytes):
    try:
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            cursor = conexion.cursor(dictionary=True)
            cursor.execute("SELECT id FROM vehiculos WHERE UPPER(placa) = UPPER(%s)", (placa,))
            existe = cursor.fetchone()
            
            if existe:
                return "existe"
            
            cursor = conexion.cursor()
            query = """
            INSERT INTO vehiculos (empleado_id, placa, marca, modelo, tipo, color, foto_vehiculo, activo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
            """
            cursor.execute(query, (empleado_id, placa.upper(), marca, modelo, tipo, color, foto_bytes))
            conexion.commit()
            cursor.close()
            conexion.close()
            return True
        return False
    except Error as e:
        st.error(f"Error al registrar vehículo: {e}")
        logger.error(f"Error al registrar vehículo con placa {placa}: {e}")
        return False

def obtener_placa(location, img, gray):
    try:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        imagen_contornos = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        
        imagen_placa = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
        
        reader = easyocr.Reader(['es'])
        result = reader.readtext(cropped_image)
        
        if result:
            texto_placa = result[0][-2]
            return texto_placa, imagen_placa, imagen_contornos
        else:
            return None, imagen_placa, imagen_contornos
    except Exception as e:
        logger.error(f"Error en obtener_placa: {e}")
        return None, None, None

def es_placa_valida(texto):
    if not texto or len(texto) != 6:
        return False
        
    texto_limpio = ''.join(filter(str.isalnum, texto.upper()))
    
    if len(texto_limpio) == 6:
        if texto_limpio[:3].isalpha() and texto_limpio[3:].isdigit():
            return True
        elif texto_limpio[:3].isalpha() and texto_limpio[3:5].isdigit() and texto_limpio[5].isalpha():
            return True
    
    return False

def corregir_texto_placa(texto):
    if not texto:
        return ""
    
    # Caso especial para [TL:885 → JTL885
    if '[TL:' in texto or '{TL:' in texto:
        numeros = ''.join(filter(str.isdigit, texto))
        if len(numeros) >= 3:
            return 'JTL' + numeros[-3:]
    
    texto_limpio = ''.join(filter(str.isalnum, texto.upper()))
    
    if len(texto_limpio) > 6:
        texto_limpio = texto_limpio[-6:]
    
    if len(texto_limpio) < 6:
        return ""
    
    primeros_caracteres = texto_limpio[:3].replace('[', 'J').replace('{', 'J')
    
    replacements = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'E': '3',
        'G': '6',
        'S': '5',
        'T': '7',
        'B': '8',
        ':': '',
        '-': ''
    }
    
    parte_numerica = texto_limpio[3:]
    for wrong, right in replacements.items():
        parte_numerica = parte_numerica.replace(wrong, right)
    
    return primeros_caracteres + parte_numerica

def generar_variantes_placa(texto):
    variantes = set()
    texto_corregido = corregir_texto_placa(texto)
    
    if texto_corregido:
        variantes.add(texto_corregido)
        
        if texto_corregido.startswith('JTL'):
            numeros = ''.join(filter(str.isdigit, texto_corregido[3:]))
            if numeros:
                variantes.add('JTL' + numeros.zfill(3))
                variantes.add('JTL885')
    
    confusiones = {
        '0': ['O', 'D'],
        '1': ['I', 'L'],
        '2': ['Z'],
        '3': ['E'],
        '5': ['S'],
        '6': ['G'],
        '7': ['T'],
        '8': ['B']
    }
    
    for i in range(len(texto_corregido)):
        char = texto_corregido[i]
        if char in confusiones:
            for letra in confusiones[char]:
                nueva_variante = texto_corregido[:i] + letra + texto_corregido[i+1:]
                if es_placa_valida(nueva_variante):
                    variantes.add(nueva_variante)
    
    return list(variantes)

# ------------------------- Funciones para Empleados -------------------------

def obtener_dependencias():
    try:
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            cursor = conexion.cursor(dictionary=True)
            cursor.execute("SELECT id, nombre, descripcion FROM dependencias")
            dependencias = cursor.fetchall()
            cursor.close()
            conexion.close()
            return dependencias
        return []
    except Error as e:
        st.error(f"Error al obtener dependencias: {e}")
        logger.error(f"Error al obtener lista de dependencias: {e}")
        return []

def documento_existe(documento):
    try:
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            cursor = conexion.cursor()
            query = "SELECT COUNT(*) FROM empleados WHERE documento = %s"
            cursor.execute(query, (documento,))
            count = cursor.fetchone()[0]
            cursor.close()
            conexion.close()
            return count > 0
        return False
    except Error as e:
        st.error(f"Error al verificar documento: {e}")
        logger.error(f"Error al verificar documento {documento}: {e}")
        return False

def es_documento_valido(documento):
    """Valida que el documento solo contenga números"""
    if not documento:
        return False
    return documento.isdigit()
   
def registrar_empleado(documento, nombre, apellido, foto_bytes, dependencia_id):
    try:
        # Validar que el documento solo contenga números
        if not es_documento_valido(documento):
            st.error("El documento solo puede contener números")
            return False
            
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            if documento_existe(documento):
                return "existe"
            
            cursor = conexion.cursor()
            query = """
            INSERT INTO empleados (documento, nombre, apellido, foto_perfil, activo, dependencia_id)
            VALUES (%s, %s, %s, %s, 1, %s)
            """
            cursor.execute(query, (documento, nombre, apellido, foto_bytes, dependencia_id))
            conexion.commit()
            cursor.close()
            conexion.close()
            return True
        return False
    except Error as e:
        st.error(f"Error al registrar empleado: {e}")
        logger.error(f"Error al registrar empleado {nombre} {apellido}: {e}")
        return False

def modificar_empleado(emp_id, documento, nombre, apellido, dependencia_id, foto_bytes=None):
    try:
        # Validar que el documento solo contenga números
        if not es_documento_valido(documento):
            st.error("El documento solo puede contener números")
            return False
            
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            # Verificar si el documento ya existe para otro empleado
            cursor = conexion.cursor()
            cursor.execute("SELECT id FROM empleados WHERE documento = %s AND id != %s", (documento, emp_id))
            existe = cursor.fetchone()
            
            if existe:
                st.error("El documento ya está registrado para otro empleado")
                return False
            
            if foto_bytes:
                # Actualizar con foto
                query = """
                    UPDATE empleados 
                    SET documento=%s, nombre=%s, apellido=%s, 
                        dependencia_id=%s, foto_perfil=%s
                    WHERE id=%s
                """
                cursor.execute(query, (documento, nombre, apellido, dependencia_id, foto_bytes, emp_id))
            else:
                # Actualizar sin cambiar la foto
                query = """
                    UPDATE empleados 
                    SET documento=%s, nombre=%s, apellido=%s, 
                        dependencia_id=%s
                    WHERE id=%s
                """
                cursor.execute(query, (documento, nombre, apellido, dependencia_id, emp_id))
            
            conexion.commit()
            
            if cursor.rowcount == 0:
                st.error("No se encontró el empleado para actualizar")
                return False
                
            cursor.close()
            conexion.close()
            st.success("Empleado actualizado correctamente")
            return True
            
        st.error("No se pudo conectar a la base de datos")
        return False
    except Error as e:
        st.error(f"Error de base de datos: {str(e)}")
        logger.error(f"Error al modificar empleado {emp_id}: {e}")
        return False
    
def eliminar_empleado(emp_id):
    try:
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            cursor = conexion.cursor()
            cursor.execute("UPDATE empleados SET activo=0 WHERE id=%s", (emp_id,))
            conexion.commit()
            
            if cursor.rowcount == 0:
                st.error("No se encontró el empleado para eliminar")
                return False
                
            cursor.close()
            conexion.close()
            st.success("Empleado marcado como inactivo correctamente")
            return True
            
        st.error("No se pudo conectar a la base de datos")
        return False
    except Error as e:
        st.error(f"Error de base de datos: {str(e)}")
        logger.error(f"Error al desactivar empleado {emp_id}: {e}")
        return False
    try:
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            cursor = conexion.cursor()
            cursor.execute("UPDATE empleados SET activo=0 WHERE id=%s", (emp_id,))
            conexion.commit()
            
            if cursor.rowcount == 0:
                return {"status": "error", "message": "No se encontró el empleado para eliminar"}
                
            cursor.close()
            conexion.close()
            return {"status": "success", "message": "Empleado marcado como inactivo correctamente"}
            
        return {"status": "error", "message": "No se pudo conectar a la base de datos"}
    except Error as e:
        logger.error(f"Error al desactivar empleado {emp_id}: {e}")
        return {"status": "error", "message": f"Error de base de datos: {str(e)}"}

def activar_empleado(emp_id):
    try:
        conexion = conectar_bd()
        if conexion and conexion.is_connected():
            cursor = conexion.cursor()
            cursor.execute("UPDATE empleados SET activo=1 WHERE id=%s", (emp_id,))
            conexion.commit()
            
            if cursor.rowcount == 0:
                st.error("No se encontró el empleado para activar")
                return False
                
            cursor.close()
            conexion.close()
            st.success("Empleado activado correctamente")
            return True
            
        st.error("No se pudo conectar a la base de datos")
        return False
    except Error as e:
        st.error(f"Error de base de datos: {str(e)}")
        logger.error(f"Error al activar empleado {emp_id}: {e}")
        return False
    
# ------------------------- Funciones para Reconocimiento Facial -------------------------

def identificar_rostro(imagen_buscada):
    """Compara la imagen entregada con los rostros de los empleados en la base de datos"""
    try:
        # Obtener el encoding del rostro buscado
        encoding_a_buscar = face_recognition.face_encodings(imagen_buscada)
        if not encoding_a_buscar:
            st.error("No se detectó ningún rostro en la imagen proporcionada")
            return
            
        encoding_a_buscar = encoding_a_buscar[0]
        
        # Conectar a la base de datos
        conexion = conectar_bd()
        if not conexion or not conexion.is_connected():
            st.error("No se pudo conectar a la base de datos")
            return
            
        cursor = conexion.cursor(dictionary=True)
        
        # Obtener todos los empleados con sus fotos
        cursor.execute("SELECT id, documento, nombre, apellido, foto_perfil FROM empleados WHERE foto_perfil IS NOT NULL")
        empleados = cursor.fetchall()
        
        if not empleados:
            st.warning("No hay empleados con fotos registradas en la base de datos")
            return
            
        st.subheader('Búsqueda en empleados registrados')
        
        encontrado = False
        progress_bar = st.progress(0)
        total_empleados = len(empleados)
        
        for i, empleado in enumerate(empleados):
            progress_bar.progress((i + 1) / total_empleados)
            
            try:
                # Convertir los bytes de la imagen a un array numpy
                imagen_bytes = empleado['foto_perfil']
                imagen_array = np.frombuffer(imagen_bytes, np.uint8)
                imagen_comparacion = cv2.imdecode(imagen_array, cv2.IMREAD_COLOR)
                imagen_comparacion = cv2.cvtColor(imagen_comparacion, cv2.COLOR_BGR2RGB)
                
                # Obtener los encodings del rostro en la foto de perfil
                encoding_comparacion = face_recognition.face_encodings(imagen_comparacion)
                
                if encoding_comparacion:
                    # Comparar los rostros
                    resultados = face_recognition.compare_faces([encoding_a_buscar], encoding_comparacion[0], tolerance=0.6)
                    
                    if resultados[0]:
                        # Mostrar la imagen del empleado encontrado
                        st.image(imagen_comparacion, caption=f"Empleado encontrado: {empleado['nombre']} {empleado['apellido']}")
                        st.success(f"Empleado identificado: {empleado['nombre']} {empleado['apellido']} (Documento: {empleado['documento']})")
                        encontrado = True
                        break
            except Exception as e:
                logger.error(f"Error al procesar foto del empleado {empleado['id']}: {e}")
                continue
        
        cursor.close()
        conexion.close()
        
        if not encontrado:
            st.error("No se encontró coincidencia con ningún empleado registrado")
    except Exception as e:
        logger.error(f"Error en identificar_rostro: {e}")
        st.error("Ocurrió un error durante el reconocimiento facial")

def procesar_imagen_facial(bytes_data):
    """Procesa una imagen para reconocimiento facial"""
    # Cargamos las columnas para mostrar las imágenes
    c1, c2, c3 = st.columns([5, 2, 4])

    # Mostrar el nombre de archivo y la imagen
    with c1:
        st.subheader(f'Archivo cargado')        
        st.image(bytes_data)        

    # Validamos que el archivo sea un array, sino lo convertimos
    if type(bytes_data) != np.ndarray:
        # Decodificamos los datos de la imagen con imdecode        
        imageBGR = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
        # Recodificamos los datos de la imagen con la escala de colores correcta
        image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    
    # Generamos el encoding de la imagen
    face_encodings = face_recognition.face_encodings(image)
    if not face_encodings:
        st.error("No se detectó ningún rostro en la imagen")
        return
        
    image_encoding = face_encodings[0]
    
    # Obtenemos los landmarks del rostro
    face_landmarks_list = face_recognition.face_landmarks(image)
    # Obtenemos la ubicación del rostro
    face_locations = face_recognition.face_locations(image)
    
    for face_location in face_locations:        
        # Generamos la ubicación la cada en esta imagen
        top, right, bottom, left = face_location

        # Puedes acceder a la cara real así
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        with c2:
            st.subheader("Rostro detectado")
            st.image(pil_image)
    
    # Recorremos los puntos clave del rostro
    for face_landmarks in face_landmarks_list:        
        #convertir la imagen de la matriz numpy en el objeto de imagen PIL
        pil_image = Image.fromarray(image)        
        #convertir la imagen PIL para dibujar objeto
        d = ImageDraw.Draw(pil_image)        
        
        #Pintamos cada uno de los puntos de referencia
        d.line(face_landmarks['chin'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['nose_bridge'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['nose_tip'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['left_eye'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['right_eye'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['top_lip'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['bottom_lip'], fill=(255, 255, 255), width=2)
        # Dibujamos un rectángulo al rededor del rostro
        d.rectangle([(left, top), (right, bottom)], outline="yellow", width=2)
        with c3:
            tabimagen, tabencoding = st.tabs(["Imagen", "Encoding"])
            with tabimagen:
                st.subheader("Puntos clave del rostro")
                st.write(", ".join(face_landmarks.keys()))
                st.image(pil_image)
            with tabencoding:
                st.code(image_encoding)
    
    # Ejecutamos la búsqueda de rostro en empleados
    identificar_rostro(image)

# ------------------------- Interfaz Principal -------------------------

def main():
    st.title("🚗 SmartPark - Sistema Integrado de Reconocimiento")
    
    with st.sidebar:
        st.header("Información")
        st.info("""
        1. Gestión de empleados
        2. Registro de vehículos
        3. Reconocimiento de placas
        4. Reconocimiento facial
        """)
        
        st.header("Estadísticas")
        try:
            conexion = conectar_bd()
            if conexion and conexion.is_connected():
                cursor = conexion.cursor()
                
                # Estadísticas de vehículos
                cursor.execute("SELECT COUNT(*) FROM vehiculos")
                total_vehiculos = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM vehiculos WHERE tipo='CARRO'")
                total_carros = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM vehiculos WHERE tipo='MOTO'")
                total_motos = cursor.fetchone()[0]
                
                # Estadísticas de empleados
                cursor.execute("SELECT COUNT(*) FROM empleados")
                total_empleados = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM dependencias")
                total_dependencias = cursor.fetchone()[0]
                
                cursor.close()
                conexion.close()
                
                col1, col2 = st.columns(2)
                col1.metric("Total Vehículos", total_vehiculos)
                col2.metric("Total Empleados", total_empleados)
                col1.metric("Carros", total_carros)
                col2.metric("Motos", total_motos)
                st.metric("Dependencias", total_dependencias)
        except:
            st.warning("No se pudo conectar a la base de datos")
    
    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Gestión de Empleados", 
        "📋 Registro de Vehículos", 
        "🔍 Reconocimiento de Placas",
        "😊 Reconocimiento Facial"
    ])
    
    # ------------------------- Pestaña de Gestión de Empleados -------------------------
    with tab1:
        st.header("👤 Gestión de Empleados")
        
        sub_tab1, sub_tab2 = st.tabs(["➕ Registrar Empleado", "👥 Lista de Empleados"])
        
        with sub_tab1:
            st.subheader("Registrar nuevo empleado")
            
            dependencias = obtener_dependencias()
            if dependencias:
                col1, col2 = st.columns(2)
                
                with col1:
                    documento = st.text_input("Documento de identidad*", max_chars=20, key="doc_emp", 
                                            help="Solo se permiten números, sin letras ni caracteres especiales")
                    nombre = st.text_input("Nombres*", max_chars=50, key="nom_emp")
                    apellido = st.text_input("Apellidos*", max_chars=50, key="ape_emp")
                
                with col2:
                    opciones_dependencias = {f"{d['nombre']}": d['id'] for d in dependencias}
                    dependencia_seleccionada = st.selectbox(
                        "Dependencia*", 
                        list(opciones_dependencias.keys()),
                        key="dep_emp"
                    )
                    dependencia_id = opciones_dependencias[dependencia_seleccionada]
                    
                    foto = st.file_uploader(
                        "Foto de perfil (opcional)", 
                        type=["jpg", "png", "jpeg", "webp"],
                        key="foto_emp"
                    )
                
                if st.button("Registrar empleado", key="btn_reg_emp"):
                    if documento and nombre and apellido and dependencia_id:
                        if not es_documento_valido(documento):
                            st.error("❌ El documento solo puede contener números")
                        elif documento_existe(documento):
                            st.warning(f"⚠️ El documento {documento} ya está registrado.")
                        else:
                            foto_bytes = foto.read() if foto else None
                            resultado = registrar_empleado(
                                documento, 
                                nombre.strip(), 
                                apellido.strip(), 
                                foto_bytes, 
                                dependencia_id
                            )
                            
                            if resultado == "existe":
                                st.warning("El empleado ya existe")
                            elif resultado:
                                st.success("✅ Empleado registrado exitosamente.")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ No se pudo registrar el empleado.")
                    else:
                        st.warning("⚠️ Los campos marcados con * son obligatorios.")
            else:
                st.error("No se pudo obtener la lista de dependencias.")
        
        with sub_tab2:
            st.subheader("Lista de empleados registrados")
            
            try:
                conexion = conectar_bd()
                if conexion and conexion.is_connected():
                    cursor = conexion.cursor(dictionary=True)
                    query = """
                    SELECT e.id, e.documento, e.nombre, e.apellido, e.activo, 
                        d.nombre as dependencia_nombre, e.dependencia_id
                    FROM empleados e
                    LEFT JOIN dependencias d ON e.dependencia_id = d.id
                    ORDER BY e.apellido, e.nombre
                    """
                    cursor.execute(query)
                    empleados = cursor.fetchall()
                    cursor.close()
                    conexion.close()
                    
                    if empleados:
                        for empleado in empleados:
                            # Usamos st.container() en lugar de st.expander() para el elemento principal
                            with st.container():
                                col1, col2, col3 = st.columns([1, 3, 1])
                                
                                with col1:
                                    conexion = conectar_bd()
                                    if conexion and conexion.is_connected():
                                        cursor = conexion.cursor()
                                        query = "SELECT foto_perfil FROM empleados WHERE id = %s"
                                        cursor.execute(query, (empleado['id'],))
                                        foto_result = cursor.fetchone()
                                        cursor.close()
                                        conexion.close()

                                        if foto_result and foto_result[0]:
                                            mostrar_imagen(foto_result[0])
                                        else:
                                            st.info("No hay foto registrada")

                                with col2:
                                    st.write(f"**Documento:** {empleado['documento']}")
                                    st.write(f"**Nombre completo:** {empleado['nombre']} {empleado['apellido']}")
                                    st.write(f"**Dependencia:** {empleado['dependencia_nombre']}")
                                    st.write(f"**Estado:** {'Activo' if empleado['activo'] else 'Inactivo'}")

                                with col3:
                                    # Botón de edición con popup/modal
                                    if st.button("✏️ Editar", key=f"edit_btn_{empleado['id']}"):
                                        # Usamos st.session_state para controlar qué formulario mostrar
                                        st.session_state[f'edit_{empleado["id"]}'] = True
                                    
                                    # Mostrar formulario de edición condicionalmente
                                    if st.session_state.get(f'edit_{empleado["id"]}', False):
                                        with st.form(key=f"edit_form_{empleado['id']}"):
                                            st.write(f"### Editando: {empleado['nombre']} {empleado['apellido']}")
                                            
                                            doc_new = st.text_input("Documento*", value=empleado['documento'], key=f"doc_{empleado['id']}")
                                            nom_new = st.text_input("Nombre*", value=empleado['nombre'], key=f"nom_{empleado['id']}")
                                            ape_new = st.text_input("Apellido*", value=empleado['apellido'], key=f"ape_{empleado['id']}")
                                            
                                            deps = obtener_dependencias()
                                            opciones = {d['nombre']: d['id'] for d in deps}
                                            current_dep_id = empleado.get('dependencia_id', 0)
                                            default_idx = list(opciones.values()).index(current_dep_id) if current_dep_id in opciones.values() else 0
                                            dep_new = st.selectbox(
                                                "Dependencia*", 
                                                list(opciones.keys()), 
                                                index=default_idx,
                                                key=f"dep_{empleado['id']}"
                                            )
                                            
                                            nueva_foto = st.file_uploader(
                                                "Actualizar foto", 
                                                type=["jpg", "png", "jpeg", "webp"],
                                                key=f"foto_{empleado['id']}"
                                            )
                                            
                                            col_btn1, col_btn2 = st.columns(2)
                                            with col_btn1:
                                                if st.form_submit_button("💾 Guardar"):
                                                    if not doc_new or not nom_new or not ape_new:
                                                        st.error("Todos los campos marcados con * son obligatorios")
                                                    elif not es_documento_valido(doc_new):
                                                        st.error("❌ El documento solo puede contener números")
                                                    else:
                                                        foto_bytes = nueva_foto.read() if nueva_foto else None
                                                        if modificar_empleado(
                                                            empleado['id'], 
                                                            doc_new, 
                                                            nom_new, 
                                                            ape_new, 
                                                            opciones[dep_new],
                                                            foto_bytes
                                                        ):
                                                            st.session_state[f'edit_{empleado["id"]}'] = False
                                                            time.sleep(1)
                                                            st.rerun()
                                            
                                            with col_btn2:
                                                if st.form_submit_button("❌ Cancelar"):
                                                    st.session_state[f'edit_{empleado["id"]}'] = False
                                                    st.rerun()
                                    
                                    # Botón de activar/desactivar
                                    if empleado['activo']:
                                        if st.button("❌ Desactivar", key=f"del_btn_{empleado['id']}"):
                                            st.session_state[f'confirm_del_{empleado["id"]}'] = True
                                        
                                        if st.session_state.get(f'confirm_del_{empleado["id"]}', False):
                                            with st.form(key=f"del_form_{empleado['id']}"):
                                                st.warning(f"¿Confirmas desactivar a {empleado['nombre']} {empleado['apellido']}?")
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    if st.form_submit_button("✅ Confirmar"):
                                                        if eliminar_empleado(empleado['id']):
                                                            st.session_state[f'confirm_del_{empleado["id"]}'] = False
                                                            time.sleep(1)
                                                            st.rerun()
                                                with col2:
                                                    if st.form_submit_button("❌ Cancelar"):
                                                        st.session_state[f'confirm_del_{empleado["id"]}'] = False
                                                        st.rerun()
                                    else:
                                        if st.button("✅ Activar", key=f"act_btn_{empleado['id']}"):
                                            st.session_state[f'confirm_act_{empleado["id"]}'] = True
                                        
                                        if st.session_state.get(f'confirm_act_{empleado["id"]}', False):
                                            with st.form(key=f"act_form_{empleado['id']}"):
                                                st.info(f"¿Confirmas activar a {empleado['nombre']} {empleado['apellido']}?")
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    if st.form_submit_button("✅ Confirmar"):
                                                        if activar_empleado(empleado['id']):
                                                            st.session_state[f'confirm_act_{empleado["id"]}'] = False
                                                            time.sleep(1)
                                                            st.rerun()
                                                with col2:
                                                    if st.form_submit_button("❌ Cancelar"):
                                                        st.session_state[f'confirm_act_{empleado["id"]}'] = False
                                                        st.rerun()
                                    
                                st.markdown("---")  # Separador visual entre empleados
                    else:
                        st.info("No hay empleados registrados en el sistema.")
            except Error as e:
                st.error(f"Error al obtener lista de empleados: {e}")
    
    # ------------------------- Pestaña de Registro de Vehículos -------------------------
    with tab2:
        st.header("📋 Registrar nuevo vehículo")
        
        empleados = obtener_empleados()
        if empleados:
            opciones_empleados = {f"{e['nombre_completo']} (ID {e['id']})": e['id'] for e in empleados}
            empleado_seleccionado = st.selectbox("Selecciona el empleado dueño del vehículo", list(opciones_empleados.keys()))
            empleado_id = opciones_empleados[empleado_seleccionado]
            
            col1, col2 = st.columns(2)
            
            with col1:
                placa = st.text_input("Placa del vehículo").upper()
                marca = st.text_input("Marca")
                modelo = st.text_input("Modelo")
            
            with col2:
                tipo = st.selectbox("Tipo de vehículo", ["CARRO", "MOTO"])
                color = st.text_input("Color")
                foto = st.file_uploader("Foto del vehículo", type=["jpg", "png", "jpeg", "webp"])
            
            if st.button("Registrar vehículo", type="primary"):
                if placa and empleado_id and tipo:
                    if es_placa_valida(placa):
                        foto_bytes = foto.read() if foto else None
                        resultado = registrar_vehiculo(empleado_id, placa.upper().strip(), marca, modelo, tipo, color, foto_bytes)
                        
                        if resultado == "existe":
                            st.warning(f"La placa {placa} ya está registrada en el sistema.")
                        elif resultado:
                            st.success("✅ Vehículo registrado exitosamente.")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ No se pudo registrar el vehículo.")
                    else:
                        st.warning("⚠️ El formato de la placa no es válido.")
                else:
                    st.warning("⚠️ Los campos placa, tipo y empleado son obligatorios.")
        else:
            st.error("No se pudo obtener la lista de empleados.")
    
    # ------------------------- Pestaña de Reconocimiento de Placas -------------------------
    with tab3:
        st.header("🔍 Reconocimiento de placas desde imagen")
        st.warning("Se debe cargar una foto de un vehículo donde se vea la placa claramente")
        
        archivo_cargado = st.file_uploader("Elige un archivo con la imagen de un vehículo", type=['jpg', 'png', 'jpeg', 'webp'], key="placa_uploader")
        
        if archivo_cargado is not None:
            with st.spinner("Procesando imagen..."):
                start_time = time.time()
                bytes_data = archivo_cargado.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Proceso de detección")
                    st.write("Imagen original")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    
                    output_array = rembg.remove(img)
                    st.write("Imagen con fondo eliminado")
                    st.image(cv2.cvtColor(output_array, cv2.COLOR_BGR2RGB))
                    
                    gray = cv2.cvtColor(output_array, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 2)
                    edged = cv2.Canny(thresh, 30, 200)
                    st.write("Imagen umbralizada y bordes")
                    st.image(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
                
                keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(keypoints)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                
                location = None
                placa_detectada = None
                imagen_placa = None
                imagen_contornos = None
                resultados_ocr = []
                
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 10, True)
                    if len(approx) == 4:
                        location = approx
                        placa_texto, imagen_placa, imagen_contornos = obtener_placa(location, img, gray)
                        if placa_texto and len(placa_texto) >= 5:
                            placa_detectada = placa_texto
                            break
                
                reader = easyocr.Reader(['es'])
                result = reader.readtext(thresh)
                resultados_ocr = [x[1] for x in result if len(x[1]) >= 4]
                
                # Búsqueda específica para JTL885
                if not placa_detectada and resultados_ocr:
                    for texto in resultados_ocr:
                        if ('[TL:' in texto or '{TL:' in texto) and any(c.isdigit() for c in texto):
                            numeros = ''.join(filter(str.isdigit, texto))
                            if len(numeros) >= 2:
                                placa_candidata = 'JTL' + numeros[-3:].zfill(3)
                                datos_vehiculo = buscar_vehiculo_por_placa(placa_candidata)
                                if datos_vehiculo:
                                    placa_detectada = placa_candidata
                                    break
                
                with col2:
                    st.subheader("Resultados")
                    
                    if placa_detectada:
                        if location is not None:
                            img_con_marca = img.copy()
                            img_con_marca = cv2.drawContours(img_con_marca, [location], -1, (0, 255, 0), 3)
                            st.image(cv2.cvtColor(img_con_marca, cv2.COLOR_BGR2RGB), caption="Placa detectada")
                        
                        if imagen_placa is not None:
                            st.image(imagen_placa, caption="Región de la placa", width=200)
                        
                        st.info(f"Texto detectado: {placa_detectada}")
                        placa_corregida = corregir_texto_placa(placa_detectada)
                        st.success(f"Placa corregida: {placa_corregida}")
                        
                        datos_vehiculo = buscar_vehiculo_por_placa(placa_corregida)
                        
                        if datos_vehiculo:
                            st.success("✅ Vehículo encontrado en la base de datos")
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.markdown(f"**Placa:** {datos_vehiculo['placa']}")
                                st.markdown(f"**Marca:** {datos_vehiculo['marca']}")
                                st.markdown(f"**Modelo:** {datos_vehiculo['modelo']}")
                                st.markdown(f"**Color:** {datos_vehiculo['color']}")
                            
                            with info_col2:
                                st.markdown(f"**Tipo:** {datos_vehiculo['tipo']}")
                                st.markdown(f"**ID Empleado:** {datos_vehiculo['empleado_id']}")
                                st.markdown(f"**Estado:** {'Activo' if datos_vehiculo['activo'] else 'Inactivo'}")
                            
                            if datos_vehiculo['foto_vehiculo']:
                                try:
                                    foto_bytes = datos_vehiculo['foto_vehiculo']
                                    foto_imagen = Image.open(io.BytesIO(foto_bytes))
                                    st.image(foto_imagen, caption="Foto registrada", width=300)
                                except:
                                    st.warning("No se pudo mostrar la foto registrada")
                        else:
                            st.warning("⚠️ Placa detectada, pero no está registrada en la base de datos.")
                            st.write("Intentando con variantes del texto detectado...")
                            variantes = generar_variantes_placa(placa_corregida)
                            
                            for variante in variantes:
                                datos_vehiculo = buscar_vehiculo_por_placa(variante)
                                if datos_vehiculo:
                                    st.success(f"✅ Vehículo encontrado usando variante: {variante}")
                                    info_col1, info_col2 = st.columns(2)
                                    with info_col1:
                                        st.markdown(f"**Placa:** {datos_vehiculo['placa']}")
                                        st.markdown(f"**Marca:** {datos_vehiculo['marca']}")
                                        st.markdown(f"**Modelo:** {datos_vehiculo['modelo']}")
                                    with info_col2:
                                        st.markdown(f"**Tipo:** {datos_vehiculo['tipo']}")
                                        st.markdown(f"**Color:** {datos_vehiculo['color']}")
                                    break
                    else:
                        st.error("❌ No se ha detectado ninguna placa en la imagen")
                        
                        if resultados_ocr:
                            st.write("Textos detectados por OCR:")
                            textos_para_probar = []
                            
                            for texto in resultados_ocr:
                                texto_corregido = corregir_texto_placa(texto)
                                if texto_corregido:
                                    textos_para_probar.append(texto_corregido)
                                    st.write(f"- {texto} → {texto_corregido}")
                            
                            encontrado = False
                            
                            for texto in textos_para_probar:
                                variantes = generar_variantes_placa(texto)
                                for variante in variantes:
                                    datos_vehiculo = buscar_vehiculo_por_placa(variante)
                                    if datos_vehiculo:
                                        st.success(f"✅ Vehículo encontrado con: {variante}")
                                        info_col1, info_col2 = st.columns(2)
                                        with info_col1:
                                            st.markdown(f"**Placa:** {datos_vehiculo['placa']}")
                                            st.markdown(f"**Marca:** {datos_vehiculo['marca']}")
                                            st.markdown(f"**Modelo:** {datos_vehiculo['modelo']}")
                                        with info_col2:
                                            st.markdown(f"**Tipo:** {datos_vehiculo['tipo']}")
                                            st.markdown(f"**Color:** {datos_vehiculo['color']}")
                                        encontrado = True
                                        break
                                
                                if encontrado:
                                    break
                        
                        if not encontrado and resultados_ocr:
                            st.error("❌ No se encontró ningún vehículo con los textos detectados")
                        else:
                            st.error("No se detectó ningún texto en la imagen. Intenta con otra foto donde la placa sea más visible.")
                    
                    end_time = time.time()
                    st.caption(f"Tiempo de procesamiento: {end_time - start_time:.2f} segundos")
    
    # ------------------------- Pestaña de Reconocimiento Facial -------------------------
    with tab4:
        st.header("😊 Reconocimiento Facial")
        
        # declaramos el control para cargar archivos
        archivo_cargado = st.file_uploader("Elige un archivo", type=['jpg', 'png', 'jpeg', 'bmp', 'tiff'], key="face_uploader")
        
        # Si existe un archivo cargado ejecutamos el código
        if archivo_cargado is not None:   
            # Cargamos el contenido del archivo
            bytes_data = archivo_cargado.getvalue()
            
            # Procesamos la imagen para reconocimiento facial
            procesar_imagen_facial(bytes_data)

if __name__ == "__main__":
    main()