# Cloud Delegate

Servidor MCP local por `stdio` que expone `delegar_a_cloud` y `listar_modelos_cloud`. Envía una tarea autocontenida a otro CLI de agente instalado en la misma máquina y devuelve su respuesta. Está escrito en Python y usa solo la biblioteca estándar.

## Qué publica este repositorio

- `mcp/cloud-mcp.py`: servidor JSON-RPC/MCP y adaptadores de CLI.
- `mcp/test_cloud_mcp.py`: pruebas con procesos simulados; no consumen cuota ni llaman a proveedores reales.

El servidor no contiene credenciales ni instala los CLI de los proveedores. Cada CLI debe estar instalado y autenticado por su dueño en el entorno donde se ejecuta el servidor. `listar_modelos_cloud` indica las rutas configuradas en esta versión y si el ejecutable correspondiente está en `PATH`; no consulta cuotas ni garantiza que una cuenta tenga acceso a un modelo. Los nombres de modelos y opciones de los CLI pueden cambiar: verifica su compatibilidad antes de usarlos.

## Uso local

```bash
python3 mcp/test_cloud_mcp.py -q
python3 mcp/cloud-mcp.py
```

Registra el servidor en un cliente MCP con `python3` como comando y la ruta absoluta a `mcp/cloud-mcp.py` como argumento. Por ejemplo, la parte esencial de una configuración MCP es:

```json
{
  "command": "python3",
  "args": ["/ruta/absoluta/cloud-delegate/mcp/cloud-mcp.py"]
}
```

La llamada a `delegar_a_cloud` requiere `prompt`. Acepta `model`, `system`, `effort`, `cwd`, `timeout_s` y `access`. `access` vale `text` por defecto; `read` y `write` habilitan permisos más amplios en el CLI delegado. Esas opciones **no son una barrera de seguridad independiente**: el proceso hereda la configuración y el acceso del usuario que lo inicia. No expongas el servidor a clientes MCP que no controles.

La delegación encadenada tiene un límite de profundidad. `CLOUD_OFFLOAD_CODEX_HOMES` permite indicar varios directorios de configuración Codex separados por `:`; si no se fija, se usa `CODEX_HOME` o `~/.codex`. No copies directorios de autenticación al repositorio.

Las respuestas de las herramientas no enumeran rutas locales de autenticación ni el directorio de trabajo del servidor. Los errores emitidos por los CLI externos pueden incluir rutas propias de esos CLI; mantén este servidor en un entorno de confianza.

## Alcance de esta extracción

Esta versión pública incluye solo el servidor reutilizable y sus pruebas. No incorpora configuración de flota, instaladores de hosts, datos de cuentas ni el historial del repositorio operativo privado. El código se ofrece para revisión; todavía no se ha elegido una licencia de reutilización.
