from help_functions import*



training_setX, training_setY, training_set_date = \
    generate_set("C:/Users/admin/PycharmProjects/Forex/sets/close_EURUSD.csv",
                 "C:/Users/admin/PycharmProjects/Forex/sets/close_EURUSD.csv", (0, 1))




'''
x_axis = [dt.datetime.strptime(d, '%d.%m.%Y').date() for d in training_set_date]
plt.plot(x_axis, training_setY,label="EUR/USD")
#plt.axvline(5, color='k', linestyle='--')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig('all.jpg', format='jpg', dpi=1200)
plt.close()
'''