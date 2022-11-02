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
    $('.progress-bar').show();
    model = await tf.loadLayersModel('model/model.json');
    $('.progress-bar').hide();
    modelLoaded = true;
})();


$("#predict-button").click(async function () {
    if(!validate()){
        return;
    }
 	let image = $('#selected-image').get(0);
	
    // Pre-process the image
    let tensor = tf.browser.fromPixels(image)
    	.resizeNearestNeighbor([224, 224]) // change the image size
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
    	});

	$("#prediction-list").empty();
    let firstClass = 'first-class';
	predictions.forEach(function (p) {
		$("#prediction-list").append(`<li class="${firstClass} list-group-item">${p.className}: ${p.probability.toFixed(2)}%</li>`);
        firstClass = '';
    });
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