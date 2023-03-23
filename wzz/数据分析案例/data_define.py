"""
数据定义的类
"""

class Record:

    def __init__(self,d,o,m,p):
        self.date = d        # 订单日期
        self.order_id = o    # 订单id
        self.money = m       # 订单金额
        self.province = p    # 订单省份

    def __str__(self):
        return f"{self.date},{self.order_id},{self.money},{self.province}"