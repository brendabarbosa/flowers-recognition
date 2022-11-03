let imageLoaded = false; 
$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	}
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
    imageLoaded = true;
});

let model;
modelLoaded = false;
(async function () {
    $('#load-model').show();
    model = await tf.loadLayersModel('model/model.json');
    $('#load-model').hide();
    modelLoaded = true;
})();


$("#predict-button").click(async function () {
    if(!validate()){
        return;
    }
    $("#load-image").show();
 	let image = $('#selected-image').get(0);
	
    let tensor = tf.browser.fromPixels(image)
    	.resizeNearestNeighbor([224, 224]) 
    	.toFloat()
    	.expandDims();
        
    let predictions = await model.predict(tensor).data();
    predictions = Array.from(predictions)
    	.map(function (p, i) {
    		return {
    			probability: p*100,
    			className: FLOWERS_CLASSES[i] 
    		};
    	}).sort(function (a, b) {
    		return b.probability - a.probability;
    	})

    $('#resultado div').html(predictions[0].className)
    $("#load-image").hide();

});


function validate(){
    if (!modelLoaded) { 
        Swal.fire(
            'Atenção',
            'Aguarde o modelo carregar.',
            'warning'
          )
          return false;
    }
	if (!imageLoaded) {
        Swal.fire(
            'Atenção',
            'Você precisa carregar uma imagem antes.',
            'warning'
          )
          return false;
    }
    return true;
}