var KTFormControls = function () {
    // Private functions
    
    var fileupload = function () {
        
        var dzone = document.querySelector("#m-dropzone-two").dropzone

        // Send all form data when sending files
        dzone.on('sending', function(file, xhr, formData){
            var data = $('#formsubmit').serializeArray();

            $.each(data, function(key, el) {
                formData.append(el.name, el.value);
            });    
        });

        dzone.on("success", function(file, responseText) {

            // TODO: Add modal pop up here for more info
            // window.location.href = ("/analysis/analyze")
        });

        $( "#formsubmit" ).validate({
            // define validation rules
            rules: {
                title: {
                    required: true,
                }
            },

            //display error alert on form submit  
            invalidHandler: function(event, validator) {     
                var alert = $('#kt_form_1_msg');
                alert.removeClass('kt--hide').show();
                KTUtil.scrollTop();
            },

            submitHandler: function (form) {
                dzone.processQueue(); // submit the form
            }
        });       
    }

    return {
        // public functions
        init: function() {
            fileupload(); 
        }
    };
}();

jQuery(document).ready(function() {    
    KTFormControls.init();
});
