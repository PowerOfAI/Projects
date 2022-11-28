# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 17:11:58 2022

@author: Pawan
"""

def prime_num_check(x):
    factors = []
    for num in range(1,x+1,1):
        if x%num==0 and len(factors)!=3:
            factors.append(num)
        else:
            pass
        
    
    if len(factors) == 2:
        return f'{x} is a prime number as its only positive factors are 1 & {factors[1]} itself.\n\nRef: For a number to be called as a prime number, it must have only two positive factors.'
    elif len(factors) == 1:
        return f'{x} is not a prime number as its only positive factor is {factors}.\n\nRef: For a number to be called as a prime number, it must have only two positive factors. Now, for 1, the number of positive divisors or factors is only one i.e. 1 itself. So, number one is not a prime number.'
    elif len(factors) == 0:
        return f"{x} is not a prime number as it doesn't have any positive factor.\n\nRef: For a number to be called as a prime number, it must have only two positive factors. Zero is neither prime nor composite. Since any number times zero equals zero, there are an infinite number of factors for a product of zero. A composite number must have a finite number of factors. One is also neither prime nor composite."

    else:
        return f'{x} is not a prime number as its atleast 3 positive factors are {factors}.\n\nRef: For a number to be called as a prime number, it must have only two positive factors, i.e. 1 and the number itself.'         


def prime_num_finder(starting_num, end_num):
    
        
    if type(starting_num)!= int:
        return f"Invalid input. The input type you entered is:\nstarting_num = {type(starting_num)}\nend_num = {type(end_num)}\nPlease enter a range of integers only."
    elif type(end_num) != int:
        return f"Invalid input. The input type you entered is:\nstarting_num = {type(starting_num)}\nend_num = {type(end_num)}\nPlease enter a range of integers only."
    elif end_num < starting_num:
        return "You seem to have entered the inputs in wrong order"
    elif end_num == starting_num:
        return f"You entered the same number i.e. {starting_num} in both the fields. Please enter values as requested"
    
    else:
        pn = []
        for x in range(starting_num,end_num+1,1):  
            factors = []
            for num in range(1,x+1,1):
                if x%num==0 and len(factors)!=3:
                    factors.append(num)
                else:
                    pass
                
            
            if len(factors) == 2:
                pn.append(x)
                          
            else:
                pass
            
        if len(pn)>=2:
            return f'There are total {len(pn)} prime numbers between {starting_num} and {end_num}. These are {pn}'
        elif len(pn)==1:
            return f'The only prime number between {starting_num} and {end_num} is {pn}'
        elif len(pn)==0:
            return "Oh ohh...the range you entered seems to be too short. There is no prime number in this range."

      
        else:
            pass

from flask import Flask, render_template, request


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("prime_index.html")

@app.route('/prime_num_checker', methods=['GET', 'POST'])
def prime_num_checker():
    if request.method =='POST':
        data = [request.form.get("aa")]
        value = int(data[0])
        result = 'Result: '+ prime_num_check(value)
        return render_template("prime_index.html", result=result)

@app.route('/prime_number_finder', methods=['GET', 'POST'])
def prime_number_finder():
    if request.method =='POST':
        var1 = [request.form.get("var1")]
        var2 = [request.form.get("var2")]
        var1 = int(var1[0])
        var2 = int(var2[0])
        result2 = 'Result: '+ prime_num_finder(var1, var2)
        return render_template("prime_index.html", result2=result2)

        


if __name__ == "__main__":
    app.run(debug=True)