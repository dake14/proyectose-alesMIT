import wfdb

print("Descargando base de datos MIT-BIH...")
# LÍNEA CORRECTA
wfdb.dl_database('mitdb', dl_dir='mitdb_data')
print("Descarga completa.")
