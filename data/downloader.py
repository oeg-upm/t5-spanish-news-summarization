import owncloud

oc = owncloud.Client('https://delicias.dia.fi.upm.es/nextcloud/')

oc.login('avogel', 'ADR.adv.291')


oc.get_file('/DataXLSum/spanish_XLSum_v2.0.tar.bz2', 'spanish_XLSum_v2.0.tar.bz2')

