from ftplib import FTP

ftp = FTP('ftp.datasus.gov.br')
ftp.login()
items = ftp.nlst('dissemin/publicos/SINAN/DADOS/FINAIS')
chik = [i for i in items if 'CHIKBR22' in i]
print("CHIK:", chik)
ftp.quit()
