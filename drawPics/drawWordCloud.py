from wordcloud import WordCloud
import matplotlib.pyplot as plt


imp_dic = {}
imp_list = [('opiate', 905), ('opioid', 156), ('drug', 112), ('heroin', 104), ('black', 91), ('dose', 90), ('addiction', 64), ('high', 58), ('day', 50), ('im', 45), ('methadone', 43), ('opium', 42), ('fentanyl', 38), ('hour', 37), ('addict', 37), ('first', 36), ('morphine', 35), ('tolerance', 31), ('time', 30), ('year', 30), ('receptor', 29), ('alcohol', 28), ('4', 26), ('xanax', 24), ('super', 22), ('addicted', 22),  ('week', 21), ('use', 21), ('opioids', 21), ('last', 20), ('withdrawal', 18), ('mdma', 18), ('pill', 17), ('codeine', 17), ('mg', 17), ('anxiety', 16), ('cocaine', 15), ('world', 15), ('5', 15), ('also', 14), ('month', 14), ('tar', 13), ('depression', 13), ('china', 13), ('lsd', 13), ('caffeine', 12), ('mixed', 12), ('stimulant', 12), ('nausea', 12), ('iv', 12), ('game', 12), ('chocolate', 12), ('pain', 11), ('meth', 11), ('gum', 11), ('benzodiazepine', 11), ('nicotine', 11), ('substance', 11), ('gram', 11), ('euphoria', 11), ('heavy', 11), ('1', 11), ('used', 10), ('u', 10), ('mild', 10), ('school', 10), ('white', 10),
            ('called', 9), ('straight', 9), ('low', 9), ('lol', 9), ('mixing', 9), ('two', 9), ('test', 9), ('minute', 9), ('since', 9), ('ton', 8), ('one', 8), ('light', 8), ('painkiller', 8), ('strong', 8), ('agonist', 8), ('acid', 8), ('depressant', 8), ('amphetamine', 8), ('overdose', 8), ('medication', 8), ('lucid', 8), ('cool', 8), ('powder', 8), ('racist', 8), ('sub', 8), ('night', 7), ('effect', 7), ('metal', 7), ('made', 7), ('thanks', 7), ('cup', 7), ('vivid', 7), ('negative', 7), ('therapy', 7), ('euphoric', 7), ('intense', 7), ('juice', 7), ('dope', 7), ('med', 7), ('prescribed', 7), ('oral', 6), ('daily', 6), ('ketamine', 6), ('oxycodone', 6), ('smoke', 6), ('sugar', 6), ('racism', 6), ('heat', 6), ('active', 6), ('6', 6), ('top', 6), ('second', 6), ('seed', 6), ('addictive', 6), ('due', 6), ('mix', 6), ('syrup', 6), ('10', 6), ('got', 6), ('ill', 6), ('pas', 5), ('wtf', 5), ('long', 5), ('took', 5), ('music', 5), ('cold', 5), ('marijuana', 5), ('using', 5), ('self', 5), ('point', 5), ('get', 5), ('synthetic', 5), ('valium', 5), ('half', 5), ('pharmaceutical', 5), ('sound', 5), ('dream', 5), ('play', 5), ('free', 5), ('milk', 5), ('7', 5), ('12', 5), ('8', 5), ('well', 5), ('class', 5), ('smoking', 5), ('cannabis', 5)]
for i in range(len(imp_list)):
    imp_dic[imp_list[i][0]] = imp_list[i][1]

wordcloud = WordCloud(width=800, height=400, max_words=200, background_color="white").generate_from_frequencies(imp_dic)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()