import owncloud

oc = owncloud.Client('https://delicias.dia.fi.upm.es/nextcloud/')

oc.login('user', 'password')


oc.get_file('/spanish_XLSum_v2.0.tar', '/spanish_XLSum_v2.0.tar')

