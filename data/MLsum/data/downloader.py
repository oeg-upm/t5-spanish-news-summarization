import owncloud

oc = owncloud.Client('https://delicias.dia.fi.upm.es/nextcloud/')

oc.login('avogel', 'ADR.adv.291')


oc.get_file('/DataMLSum/es_test.txt', 'es_test.txt')
oc.get_file('/DataMLSum/es_train.txt', 'es_train.txt')
oc.get_file('/DataMLSum/es_val.txt', 'es_val.txt')