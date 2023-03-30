#计算三角形面积
def getarea(x1,y1,x2,y2,x3,y3):
    return abs(x1*y2-x1*y3+x2*y3-x2*y1+x3*y1-x3*y2)/2

#把小的三角形作为分割模板(我也不知道是否必要但是我觉得应该这么做)
def getmintriangle(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6):
    if getarea(x1,y1,x2,y2,x3,y3)>getarea(x4,y4,x5,y5,x6,y6):
        return x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6
    else:
        return x4,y4,x5,y5,x6,y6,x1,y1,x2,y2,x3,y3

#判断点实在模板得内外侧，好决定运用哪种情况可以参考Sutherland-hodgman算法
def judgeinorout(x1,y1,x2,y2,x3,y3,x4,y4):#(x1,y1)(x2,y2)为直线，(x3,y3)为模板三角形得顶点，有顶点为内侧，判断(x4,y4)在内外侧
    if(x1==x2 and y1!=y2):       #模板(0, -1)(0.5, 1)直线，顶点(1, -1)，判断点(0.6, 0.6)，(0, 0)
        crossx = x1             #k = 4   crossx =
        crossy = y3
    elif y1==y2 and x1!=x2:
        crossx = x3
        crossy = y1
    else:
        k = (y2-y1)/((x2-x1))             #true代表在内测，false代表在外侧
        crossx = (k*x1+x3/k-y1+y3)/(k+1/k)
        # if(crossx<0.000001):
        #     crossx = 0
        crossy = k*(crossx-x1)+y1
        # if(crossy<0.000001):
        #     crossy = 0
    #算cos角x3crossxx4得正负判断内外侧
    crossxx3 = (crossy-y3)**2+(crossx-x3)**2
    crossxx4 = (crossy-y4)**2+(crossx-x4)**2
    x3x4 = (y3-y4)**2+(x3-x4)**2
    #print(crossxx3+crossxx4-x3x4)
    if(crossxx3+crossxx4-x3x4>=0):
        return [True,0,0]
    else:
        return [False,crossx,crossy]

#计算两个三角形得相交面积
def getcrosstrianglearea(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6):
    px1,py1,px2,py2,px3,py3,px4,py4,px5,py5,px6,py6 = getmintriangle(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6)
    tri1 = [(px1,py1),(px2,py2),(px3,py3)]
    print(tri1)
    tri2 = [(px4,py4),(px5,py5),(px6,py6),(px4,py4)]
    print(tri2)
    #tri1面积更小
    tri3 = judgeedge(tri2,tri1[0][0],tri1[0][1],tri1[1][0],tri1[1][1],tri1[2][0],tri1[2][1])
    print(tri3)
    tri4 = judgeedge(tri3,tri1[1][0],tri1[1][1],tri1[2][0],tri1[2][1],tri1[0][0],tri1[0][1])
    print(tri4)
    pout = judgeedge(tri4,tri1[0][0],tri1[0][1],tri1[2][0],tri1[2][1],tri1[1][0],tri1[1][1])
    print(pout)

#计算两条直线相交交点
def linecrosspo(x1,y1,x2,y2,x3,y3,x4,y4):
    if (x1 == x2 and y1 != y2 and y3!=y4):
        x = x1
        y = (y3-y4)/(x3-x4)*(x-x3)+y3
    elif y1 == y2 and x1 != x2 and x3!=x4:
        y = y1
        x = (y-y3)/((y3-y4)/(x3-x4))+x3
    elif x3 == x4 and y3!=y4 and y1!=y2:
        x = x3
        y = (y1-y2)/(x1-x2) * (x - x1) + y1
    elif y3 == y4 and x3 != x4 and x1!=x2:
        y = y3
        x = (y - y1) / ((y1-y2)/(x1-x2)) + x1
    elif x1 == x2 and y3 == y4:
        x = x1
        y = y3
    elif x3 == x4 and y1 == y2:
        x = x3
        y = y1
    else:
        k1 = (y3-y4)/(x3-x4)
        k2 = (y1-y2)/(x1-x2)
        x = (k1*x4-k2*x1+y1-y4)/(k1-k2)
        # if (x < 0.000001):
        #     x = 0
        y = k1*(x-x4)+y4
    return x,y

#判断是SH算法中的哪种情况
def judgeedge(tri2,x1,y1,x2,y2,x3,y3):
    poin = []
    for i in range(len(tri2)-1):
        if(judgeinorout(x1,y1,x2,y2,x3,y3,tri2[i][0],tri2[i][1])[0]==True and judgeinorout(x1,y1,x2,y2,x3,y3,tri2[i+1][0],tri2[i+1][1])[0]==True):
            poin.append((tri2[i+1][0],tri2[i+1][1]))
            #print('a')
        elif(judgeinorout(x1,y1,x2,y2,x3,y3,tri2[i][0],tri2[i][1])[0]==True and judgeinorout(x1,y1,x2,y2,x3,y3,tri2[i+1][0],tri2[i+1][1])[0]==False):
            poin.append((linecrosspo(x1,y1,x2,y2,tri2[i][0],tri2[i][1],tri2[i+1][0],tri2[i+1][1])))
            #print('b')
        elif(judgeinorout(x1,y1,x2,y2,x3,y3,tri2[i][0],tri2[i][1])[0]==False and judgeinorout(x1,y1,x2,y2,x3,y3,tri2[i+1][0],tri2[i+1][1])[0]==True):
            poin.append((linecrosspo(x1,y1,x2,y2,tri2[i][0],tri2[i][1],tri2[i+1][0],tri2[i+1][1])))
            poin.append((tri2[i+1][0],tri2[i+1][1]))
            #print('c')
    if(len(poin)==0 or (poin[0][0]-poin[len(poin)-1][0])**2+(poin[0][1]-poin[len(poin)-1][1])**2<0.001):
        return poin
    poin.append((poin[0][0],poin[0][1]))
    return poin

#测试代码
getcrosstrianglearea(0,5,1,-1,-1,-1,0,3,2,-2,-2,-2)
