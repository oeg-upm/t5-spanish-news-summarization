import owncloud

oc = owncloud.Client('https://delicias.dia.fi.upm.es/nextcloud/')

oc.login('avogel', 'ADR.adv.291')


oc.get_file('/T5esp/espt5-large.rar', 'espt5-large.rar')

