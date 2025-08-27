Re=80
terminal_output_name='ciao.txt'
DRAG=[12,11,10]
LIFT=[0,2,3,4]
with open(terminal_output_name, "w") as txt:
                txt.write(f'''Drag evolution: 
                    First calculation, Drag: {DRAG[0]}
                    Mid-way calculation, Drag: {DRAG[len(DRAG)//2]}
                    Final calculation, Drag: {DRAG[-1]} \n''')
                txt.write(f'''Lift evolution: 
                    First calculation, Lift: {LIFT[0]}
                    Mid-way calculation, Lift: {LIFT[len(LIFT)//2]}
                    Final calculation, Lift: {LIFT[-1]}
                    Variabili importanti relative all'esecuzione''')
                
print('codice eseguito!')