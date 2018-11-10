import sparsity as sp
import sparsityh as sph
import PCA as pca
import NMF as nmf
import projectiveNMF as pnmf







#sparsity:

pca_  = sp.sparsity(pca.dataPCA)
nmf_  = sp.sparsity(nmf.data_NMF)
pnmf_ = sp.sparsity(pnmf.w)

print("\nSparsity :")
print("PCA : ",pca_)
print("NMF : ",nmf_)
print("P-NMF : ",pnmf_)

#sparsityH:


Spca_ , Apca_  = sph.sparsityH(pca.dataPCA)
Snmf_ ,Anmf_ = sph.sparsityH(nmf.data_NMF)
Spnmf_,Apnmf_ = sph.sparsityH(pnmf.w)


#print(pca.dataPCA[1].reshape(32,1)[2])

print("\nSparsityH :")
print("PCA : ",Spca_)
print("NMF : ",Snmf_)
print("P-NMF : ",Spnmf_)