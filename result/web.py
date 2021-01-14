from multiprocessing import Pool

from flask import Flask, render_template, redirect, request, url_for, json, session
import dbModule
import aligned
import diffuse

app = Flask(__name__, static_url_path='/static')

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


# db query###################

# insert --> 회원가입, select --> 로그인
#회원 가입 과정
def insert(owner, password, front_location, align_location, diffuse_location, filename):
    db_class = dbModule.Database()
    sql = "INSERT INTO graduateDB.user (owner, password,front_location, align_location, diffuse_location, filename) VALUES ('%s','%s','%s','%s','%s','%s')" % (owner, password,front_location, align_location, diffuse_location, filename)

    db_class.execute(sql)
    db_class.commit()

#아이디와 비밀번호가 동일한 유저를 찾음 --> 해당 사용자의 이미지를 볼 수 있음
def select(owner):
    db_class = dbModule.Database()

    sql = "SELECT * FROM graduateDB.user WHERE owner='%s'" % (owner)
    print(sql)
    row = db_class.executeAll(sql)
    db_class.commit()
    print(row[0])

    # save session of image_location
    if row is None:
        return "no data in DB"
    return row[0]


def selectAll():
    db_class = dbModule.Database()

    sql = "SELECT * FROM graduateDB.user"
    row = db_class.executeAll(sql)

    return row


def password_check(owner, password):
    db_class = dbModule.Database()

    sql = "SELECT * FROM graduateDB.user WHERE owner='%s' AND password='%s'" % (owner,password)
    print(sql)
    row = db_class.executeAll(sql)
    db_class.commit()
    print(row[0])

    # save session of image_location
    if row is None:
        return "no data in DB"


@app.route("/convert", methods=['POST'])
def gan_running():
    align_location = session['align_location']
    diffuse_location = session['diffuse_location']
    filename = session['filename']
    hair_style = request.form['hair_style_input']
    print("hair_style is"+ hair_style)
    print(align_location)
    print(diffuse_location)
    print("-;-;-;-;-;-;-;-;-;-")
    print(filename)
    print("-------------")
    print("Gan is running")
    pool.apply_async(encoder, args=(align_location, diffuse_location, filename, hair_style,))
    return redirect(url_for('index_main_page'))


def image_align_page(front_location, align_location, filename):
    print("Gan is running in apply_async!")

    print("Gan is running!")
    # B_to_A_2.Before_to_After(dual, filename)
    #민근이 함수
    aligned.align_module(front_location, align_location, filename)
    print("Gan is finished!!")


def encoder(align_location, diffuse_location, filename, hair_style):
    diffuse.encode(align_location , hair_style, diffuse_location)

    return


############################


@app.route("/logout")
def logout():
    session.pop('owner', None)
    session.clear()
    return redirect(url_for('main_page'))


@app.route("/main_page", methods=['GET', 'POST'])
def main_page():
    return render_template('index_new.html')


@app.route("/buttons.html")
def button_page():
    return render_template('buttons.html')


@app.route("/login.html")
def login_page():
    return render_template('login.html')


@app.route("/session.html", methods=['GET', 'POST'])
def session_save():
    try:
        session['owner'] = request.form['owner']
        if session['owner'] == "":
            return redirect(url_for('main_page'))

        front_image = request.files['front_image']

        front_location = 'image/front/'
        align_location = 'image/align/'
        diffuse_location = 'image/diffuse/'
        _filename = session['owner'] + ".png"
        
        print("image location aligned!")
        session['front_location'] = front_location+_filename
        session['align_location'] = align_location+_filename
        session['diffuse_location'] = diffuse_location+_filename
        session['filename'] = _filename

        print("image saving!")
        front_image.save('static/' + session['front_location'])

        print("image saved!")

        # DB save
        insert(session['owner'], request.form['password'], front_location, align_location, diffuse_location, _filename)
        print("insert success!!")
        image_align_page(front_location, align_location, _filename)
    
        #민근이 align 실행

        return redirect(url_for('index_main_page'))

    except:  # POST 인자값이 비어있는 상태로 request_form을 하면 exception 발생
        print("exception!!")
        return redirect(url_for('main_page'))


def session_update():
    select_return = select(session['owner'])

    select_return = json.dumps(select_return)
    print(select_return)
    select_return = json.loads(select_return)

    print(select_return)
    session['owner']=select_return['owner']
    session['front_location'] = select_return['front_location']
    session['align_location'] = select_return['align_location']
    session['diffuse_location'] = select_return['diffuse_location']
    session['filename'] = select_return['filename']

@app.route("/sessionOld.html", methods=['POST'])
def session_save_old():
    try:
        session['owner'] = request.form['owner']
        if session['owner'] == "":
            return redirect(url_for('main_page'))

        password_check(request.form['owner'], request.form['password'])
        # 이름을 적지 않으면 redirect를 걸어줌

        # 이름이 비어있는 상태로 상세 페이지로 접근할때 database issue가 발생할 수 있음


        # 그렇지 않으면 session을 가지고 상세 페이지로 접근
        # DB save
        session_update()
        return redirect(url_for('index_main_page'))

    except:  # POST 인자값이 비어있는 상태로 request_form을 하면 exception 발생
        return redirect(url_for('main_page'))


@app.route("/detail.html")
def index_main_page():
    return render_template('detail.html', patient_data=session)
    # front_location, above_location to print image


@app.route("/manage_detail", methods=['POST'])
def manage_detail():
    hole_data = request.form['hole_data']
    print("123123123")
    # print(hole_data.type)
    hole_data = hole_data.replace("'", "\"")
    hole_data = hole_data.replace("None", "\"None\"")
    hole_data = json.loads(hole_data)
    print(hole_data['front_result'])

    return render_template('manage_detail.html', patient_data=hole_data)


@app.route("/repre_case.html")
def repre_case_page():
    return render_template('repre_case.html', patient_data=session['owner'])


@app.route("/index.html")
def index_page():
    return render_template('index.html')


@app.route("/charts.html")
def chart_page():
    return render_template('charts.html')


@app.route("/management")
def table_page():
    row = selectAll()

    return render_template('tables.html', patient_data=row, length=len(row))


@app.route("/cards.html")
def card_page():
    return render_template('cards.html')


@app.route("/register.html")
def register_page():
    return render_template('register.html')

host_addr = "0.0.0.0"
#host_addr = "127.0.0.1"
port_num = "8080"

dual_pool = [1]
if __name__ == "__main__":
    pool = Pool(processes=1)

    app.run(host_addr, port_num)
