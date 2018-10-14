// document.
var wage = document.getElementById("search");
wage.addEventListener("keydown", function (e) {
    if (e.keyCode === 13) {  //checks whether the pressed key is "Enter"
        validate(e);
    }
});

function validate(e) {
    var text = e.target.value;

    // console.log(text)
    $.ajax({
            url: "http://127.0.0.1:8000/intention/predict_svm?sentence=" + text,
            success: function (result) {
                var obj = JSON.parse(result);
                // console.log(obj);
                document.getElementById('info').innerHTML = "intention: "+obj['predict'] ;
                // document.getElementById('prob').innerHTML = "probability: " + obj['prob'] ;

            }
        }
    )
    //validation of the input...
}