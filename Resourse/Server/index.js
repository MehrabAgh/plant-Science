const express = require('express')
const { PythonShell } = require('python-shell');

const app = express()
const port = 3000

app.set('view engine', 'ejs');
app.use(express.static("public"));

app.get('/', (req, res) => {

    let options = {
        args: ['TOL', 'value2', 'value3']
    };

    let pyshell = new PythonShell('index.py', options);

    pyshell.send('hello');

    pyshell.on('message', function(message) {
        console.log(message);
    });
    pyshell.end(function(err, code, signal) {
        if (err) throw err;
        console.log('The exit code was: ' + code);
        console.log('The exit signal was: ' + signal);
        console.log('finished');
    });
    res.render('viewIndex', {
        title: 'bobbyhadz.com',
        message: 'Express.js example',
    });
})


app.listen(port, () => console.log(`app listening on port ${port}!`))