import AI

mbj = AI.Data('rod')

m = mbj.LoadData()

print(m)

mbj.SaveData(m)

ai = AI.AI([50,200,50],1)

ai.Farve('R')
ai.Farve('G')
ai.Farve('B')

A = ai.ai()
print(A)

l = AI.Farve_tilfojelse('R')
l.Farve([10,10,10])