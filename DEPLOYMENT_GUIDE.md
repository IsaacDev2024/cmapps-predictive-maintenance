# 🚀 GUÍA DE DESPLIEGUE - GitHub & Streamlit Cloud

Esta guía te ayudará a subir tu proyecto a GitHub y desplegarlo en Streamlit Cloud **GRATIS**.

---

## 📋 PARTE 1: SUBIR A GITHUB

### ✅ Paso 1: Crear Repositorio en GitHub

1. Ve a [GitHub.com](https://github.com) e inicia sesión
2. Haz clic en el botón **"+"** (esquina superior derecha) → **"New repository"**
3. Configura tu repositorio:
   - **Repository name**: `cmapps-predictive-maintenance` (o el nombre que prefieras)
   - **Description**: `Sistema de Mantenimiento Predictivo para Motores Jet - NASA C-MAPSS`
   - **Visibility**: 
     - ✅ **Public** (recomendado para Streamlit Cloud gratis)
     - ⚠️ Private (requiere plan pago en Streamlit)
   - **NO** marques: "Add a README file" (ya lo tenemos)
   - **NO** marques: "Add .gitignore" (ya lo tenemos)
   - Selecciona **License**: MIT License
4. Haz clic en **"Create repository"**

### ✅ Paso 2: Conectar tu Repositorio Local con GitHub

Copia el URL de tu nuevo repositorio (algo como `https://github.com/tu-usuario/cmapps-predictive-maintenance.git`)

Ejecuta estos comandos en la terminal:

```bash
# 1. Agregar el repositorio remoto
git remote add origin https://github.com/TU-USUARIO/cmapps-predictive-maintenance.git

# 2. Verificar que se agregó correctamente
git remote -v

# 3. Subir tu código a GitHub
git push -u origin main
```

**⚠️ Importante**: Si GitHub te pide autenticación:
- Usa un **Personal Access Token** en lugar de tu contraseña
- Ve a: GitHub → Settings → Developer settings → Personal access tokens → Generate new token
- Dale permisos de `repo` y copia el token generado

### ✅ Paso 3: Verificar en GitHub

1. Refresca la página de tu repositorio en GitHub
2. Deberías ver todos tus archivos, incluyendo:
   - ✅ README.md
   - ✅ Dashboard/
   - ✅ requirements.txt
   - ✅ .gitignore
   - ✅ LICENSE

---

## 🌐 PARTE 2: DESPLEGAR EN STREAMLIT CLOUD (GRATIS)

### ✅ Paso 1: Crear Cuenta en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Haz clic en **"Sign up"** o **"Continue with GitHub"**
3. Autoriza a Streamlit Cloud para acceder a tus repositorios de GitHub
4. Completa tu perfil (si es necesario)

### ✅ Paso 2: Crear Nueva App

1. En el dashboard de Streamlit Cloud, haz clic en **"New app"**
2. Completa el formulario:

   **Repository:**
   - Selecciona tu repositorio: `tu-usuario/cmapps-predictive-maintenance`
   
   **Branch:**
   - Selecciona: `main`
   
   **Main file path:**
   - ⚠️ **MUY IMPORTANTE**: Escribe: `Dashboard/app.py`
   - (No solo `app.py`, debe incluir la carpeta `Dashboard/`)
   
   **App URL (opcional):**
   - Personaliza la URL (ejemplo: `cmapps-nasa-predictive`)
   - O deja que Streamlit genere una automáticamente

3. Haz clic en **"Deploy!"**

### ✅ Paso 3: Esperar el Despliegue

Streamlit Cloud automáticamente:
1. ✅ Clonará tu repositorio
2. ✅ Instalará las dependencias de `requirements.txt`
3. ✅ Ejecutará `Dashboard/app.py`
4. ✅ Te dará una URL pública

**Tiempo estimado**: 2-5 minutos

⚠️ **Si hay errores**:
- Revisa los logs en la consola de Streamlit Cloud
- Verifica que `requirements.txt` esté en la raíz del proyecto
- Asegúrate que la ruta sea `Dashboard/app.py` (con la carpeta)

### ✅ Paso 4: Obtener tu URL

Una vez desplegado, obtendrás una URL como:
```
https://cmapps-nasa-predictive-tu-usuario.streamlit.app
```

---

## 🔧 CONFIGURACIÓN AVANZADA (OPCIONAL)

### Configurar Secrets (Variables de Entorno)

Si necesitas claves API o configuraciones sensibles:

1. En Streamlit Cloud, ve a tu app
2. Haz clic en **"⋮"** (tres puntos) → **"Settings"**
3. Ve a **"Secrets"**
4. Agrega tus secrets en formato TOML:

```toml
# Ejemplo
api_key = "tu-clave-secreta"
database_url = "tu-url-de-bd"
```

### Actualizar la App

Cada vez que hagas cambios en GitHub:

```bash
# 1. Hacer cambios en tu código local
# 2. Commit
git add .
git commit -m "Descripción de los cambios"

# 3. Push a GitHub
git push origin main
```

**Streamlit Cloud automáticamente detectará los cambios y re-desplegará tu app** 🎉

---

## 📝 ACTUALIZAR EL README CON TU URL

Una vez tengas tu URL de Streamlit Cloud, actualiza el README:

1. Abre `README.md`
2. Busca la línea que dice:
   ```markdown
   ### 🌐 **[Ver Dashboard en Vivo →](https://tu-dashboard.streamlit.app)**
   ```
3. Reemplaza `https://tu-dashboard.streamlit.app` con tu URL real
4. Guarda y haz commit:
   ```bash
   git add README.md
   git commit -m "Update: URL del dashboard desplegado"
   git push origin main
   ```

También actualiza `Dashboard/README.md` de la misma forma.

---

## ✅ CHECKLIST FINAL

Antes de compartir tu proyecto, verifica:

- ✅ El repositorio está en GitHub y es público
- ✅ El dashboard está desplegado en Streamlit Cloud
- ✅ La URL del dashboard funciona correctamente
- ✅ El README tiene la URL actualizada
- ✅ Todas las visualizaciones cargan correctamente
- ✅ El modelo LSTM hace predicciones sin errores
- ✅ Los datos se cargan correctamente

---

## 🎉 ¡LISTO!

Tu proyecto ahora está:
- ✅ **En GitHub**: Visible para el mundo, portfolio profesional
- ✅ **En Streamlit Cloud**: Dashboard interactivo accesible 24/7
- ✅ **Documentado**: README profesional y completo

### 📢 Comparte tu Proyecto:

```markdown
🚀 Sistema de Mantenimiento Predictivo para Motores Jet

Modelo LSTM con 98.5% de precisión para predecir fallos en motores.

📊 Dashboard: https://tu-url.streamlit.app
💻 GitHub: https://github.com/tu-usuario/cmapps-predictive-maintenance

#MachineLearning #DeepLearning #PredictiveMaintenance #NASA #LSTM
```

---

## 🆘 SOLUCIÓN DE PROBLEMAS COMUNES

### Error: "ModuleNotFoundError"
**Solución**: Verifica que todas las dependencias estén en `requirements.txt`

### Error: "File not found: app.py"
**Solución**: Asegúrate de usar `Dashboard/app.py` como ruta principal

### Error: "Memory limit exceeded"
**Solución**: Los modelos muy grandes pueden exceder el límite gratuito (1GB RAM)
- Considera comprimir el modelo
- O usa técnicas de model quantization

### La app se queda "loading" eternamente
**Solución**: 
- Revisa los logs en Streamlit Cloud
- Puede ser un problema con las versiones de TensorFlow
- Intenta especificar versiones exactas en `requirements.txt`

### No se ve la configuración de tema
**Solución**: 
- Verifica que `Dashboard/.streamlit/config.toml` esté en el repo
- El tema solo funciona si el archivo está en la ubicación correcta

---

## 📚 RECURSOS ADICIONALES

- [Documentación de Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [GitHub Guides](https://guides.github.com/)
- [Streamlit Forum](https://discuss.streamlit.io/)

---

**¿Necesitas ayuda?** Abre un issue en GitHub o consulta los recursos anteriores.

**¡Buena suerte con tu proyecto! 🚀**
