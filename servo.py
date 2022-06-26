import sys
import time
import serial
import PySimpleGUI as sg

def setAngle(channel : int, degree : float):
    cmd = "%d,%.1f\r" % (channel, degree)

    while True:
        try:
            n = ser.write(cmd.encode('utf-8'))
            break
        except serial.SerialTimeoutException:
            print("write time out")
            time.sleep(1)

    ret = ser.read_all().decode('utf-8')
    if "error" in ret:
        print("read", ret)

def spin(key):

    return [ 
        sg.Text(key),
        sg.Spin(list(range(0, 270 + 1)), initial_value=90, size=(5, 1), key=key, enable_events=False, bind_return_key=True )
    ]

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('COMを指定してください。')
        print('python servo.py COM*')
        sys.exit(0)

    com_port = sys.argv[1]

    try:
        ser = serial.Serial(com_port, 115200, timeout=1, write_timeout=1)
    except serial.serialutil.SerialException: 
        print(f'指定されたシリアルポートがありません。{com_port}')
        sys.exit(0)

    layout = [
        [
            spin('J1'),
            spin('J2'),
            spin('J3'),
            spin('J4'),
            spin('J5'),
            spin('J6')
        ]
        ,
        [ sg.Button('Close')]
    ]

    window = sg.Window('Servomotor', layout, disable_minimize=True, element_justification='c')

    while True:
        event, values = window.read(timeout=1)
            
        jKeys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

        if event in jKeys:
            channel = jKeys.index(event)
            degree  = float(values[event])

            setAngle(channel, degree)

        elif event == sg.WIN_CLOSED or event == 'Close':
            break

    window.close()

