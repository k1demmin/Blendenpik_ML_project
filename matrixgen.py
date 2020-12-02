'''This is the module for matrix generation'''
import numpy as np
import math
def generate_leverage_score(num_rows, li_sum, mu_li, num_li):
	li = np.zeros((num_rows))
	if (mu_li*num_li>li_sum):
		print("leverage scores sum cannot exceed rank")
		return li
	li[num_rows-num_li:]=mu_li
	li[:num_rows-num_li] = (li_sum-mu_li*num_li)/(num_rows-num_li)
	#print(li)

	return li

# need to verify solvability previously?
# method from dhillon and tropp to solve
def DiffcsSol(A, i, j, nom):
	a = A[i,:].T
	ata = a.T @ a
	
	b = A[j,:].T
	btb = b.T @ b
	
	atb = a.T @ b
	
	tmp1 = -atb/(btb-nom)
	tmp2 = (math.sqrt(atb*atb-(ata-nom)*(btb-nom)))/(btb-nom)
	
	if tmp1 <= 0: #for numerical stability
		tp = tmp1-tmp2
	else:
		tp = tmp1+tmp2
	
	c = 1/math.sqrt(1+tp*tp)
	s = c*tp
	
	return c,s
	
def generate_U(m, n, li):
	
	eps = 2.22e-16
	if len(li) != m:
		print("you need m leverage scores")
	#print(np.sum(li)-n)
	if abs(np.sum(li)-n) > 4*m*eps:
		print("leverage scores must sum to n")
	U = np.zeros((m,n))
	U = np.zeros((m,n))
	U[m-n:,:] = np.eye(n)

	a = np.zeros((m))
	a[m-n:]=1
	j=m-n
	i=m-n-1

	for ii in range (1,m,1):
		#print('step',ii,'  i, j =',i,j)
		if i<0 or j >= m:
			break;
			
		if -(a[i] - li[i]) < (a[j] - li[j]):
			c, s = DiffcsSol(U, i, j, li[i])
			U[[i,j],:] = [[c,-s],[s,c]] @ U[[i,j],:]
		else:
			c, s = DiffcsSol(U, j, i, li[j])
			U[[j,i],:] = [[c,-s],[s,c]] @ U[[j,i],:]
		
		a[i] = U[i,:] @ U[i,:].T
		a[j] = U[j,:] @ U[j,:].T
		
		if abs(a[i]-li[i]) < 10e-15: # we are done with this small-index row
			i -= 1
		if abs(a[j]-li[j]) < 10e-15: # we are done with this big-index row
			j += 1
	np.random.shuffle(U)

	return U

def matrix_generate_coherent(rank, m, n, mu, sigma2, coherent, mu_li, num_li):

	M = np.random.normal(mu, math.sqrt(sigma2), (m,n))
	U, s, V = np.linalg.svd(M,full_matrices=False) #get reduced SVD
#	print("U before")
#	for rows in U:
#		print (LA.norm(rows,2))
	if coherent == True:
#		N = np.random.normal(mu, sigma, (m,n))
#		d = math.floor(m*0.9)
#		selected_row = np.sort(np.random.permutation(m)[:d])
#		for row in selected_row:
#			N[row,:]=0
#		U,_,_ = np.linalg.svd(N,full_matrices=False) #get reduced SVD
		
		li = generate_leverage_score(m, rank, mu_li, num_li)
		#print(li)
		U = generate_U(m, rank, li)
		A = np.dot(U*s[:rank],V[:rank,:])

	else:
		A = np.dot(U[:,:rank]*s[:rank],V[:rank,:])
#	print('rank=',LA.matrix_rank(A))
#	print("U li")
#	for rows in U:
#		print (LA.norm(rows,2))
#	print("V li")
#	for cols in V[:rank,:]:
#		print (LA.norm(cols,2))
	return A
