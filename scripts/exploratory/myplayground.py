semesters = [6, 4, 7, 2, 8, 1, 5, 3]
supersenior = [9]
semesters.extend(supersenior)
sorted_semesters = sorted(semesters)
print(sorted_semesters)

momcilo_set = {'Basketball' , 'Atheist', 'Smart'}
omar_set = {'Basketball', 'Muslim', 'Dumb'}
print(omar_set.intersection(momcilo_set))

germany = 50

if germany == 114:
    print('Alles ist gut')
elif germany < 114 and germany > 40:
    print('Not bad, kid!')
elif germany == 0:
    print('Alles ist los')


