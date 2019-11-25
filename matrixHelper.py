import random
def matrixExpander(m):
    matrix=m.copy()
    #expanding horizontally
    for i in range (len(matrix)):
        matrix[i]=matrix[i]+matrix[i][::-1]
    #exexpanding vertically
    for i in range(len(matrix)):
        inverseMatrix=matrix.copy()[::-1]
    matrix+=inverseMatrix
    return matrix
def printMatrix(m):
    for i in range(len(m)):
        for j in range(len(m)):
            print(m[i][j],end=' ')
        print('\n')

def writeMatrix(matrix, f):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if j != len(matrix) - 1:
                f.write(str(matrix[i][j]) + ' ')
            else:
                f.write(str(matrix[i][j]))
        if i != len(matrix) - 1:
            f.write('\n')

def writeRandomHouses(f, maxTA, matrixOrder):
    homes=''
    numHomes=0
    brr=1
    listNums=[]
    while(len(listNums)!=99):
        num=random.randint(1,192)
        if(num not in listNums):
            listNums.append(num)
    for i in listNums:
        homes+=str(i)+' '

    """for i in range(0,matrixOrder,6):
        if(numHomes+4>=maxTA):
            break
        if(brr==1):
            homes+=str(i+2)+' '+str(i+4)+' '+str(i+5)+' '+str(i+6)+' '
            numHomes+=4
            brr+=1
        else:
            listNums=[]
            while(len(listNums)!=4):
                num=random.randint(i+1,i+6)
                if(num not in listNums):
                    listNums.append(num)
            homes+=str(listNums[0])+' '+str(listNums[1])+' '+str(listNums[2])+' '+str(listNums[3])+' '
            numHomes+=4"""
    #f.write(str(numHomes))
    f.write(str(99))
    f.write('\n')

    for i in range(1, matrixOrder+1):
        if i != matrixOrder:
            f.write(str(i)+" ")
        else:
            f.write(str(i))
            f.write('\n')
    f.write(str(homes))

def writeInputFile(matrixOrder, matrix, f, maxTA, source):
    f.write(str(matrixOrder))
    f.write('\n')
    writeRandomHouses(f, maxTA, matrixOrder)
    f.write('\n')
    f.write(str(source))
    f.write('\n')
    writeMatrix(matrix, f)








matrix12=matrixExpander([[0,10,14,12,16,28],[10,0,20,0,0,34],[14,20,0,8,12,0],[12,0,8,0,0,38],[16,0,12,0,0,18],[28,34,0,38,18,0]])
matrix24=matrixExpander(matrix12)
matrix48=matrixExpander(matrix24)
matrix96=matrixExpander(matrix48)
#
#matrix192=matrixExpander(matrix96)


for i in range(len(matrix96)):
    for j in range(len(matrix96)):
        if matrix96[i][j] == 0:
            matrix96[i][j] = 'x'

"""f = open("200xy.in", "w+")
writeInputFile(192,matrix192, f, 100, 1)
f.close()"""
def writeOutputFile(homes):
    f.write('1')
    f.write('\n')
    f.write('1')
    f.write('\n')
    f.write('1 '+homes)
homes=''
homeAddys='74 47 12 87 65 25 92 78 5 34 48 77 6 79 67 33 46 16 49 30 56 7 22 93 94 91 10 18 50 90 2 21 86 83 39 8 32 84 19 54 57 29 11 59 95 42 36 89 28'.split(' ')
for i in homeAddys:
    homes+=str(i)+' '
f=open("100xy.out","w+")
writeOutputFile(homes)
f.close()
