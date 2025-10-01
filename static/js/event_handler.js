document.addEventListener('DOMContentLoaded', domReady);
    let dicsStatic = null;
    let dicsDynamic = null;
        function domReady() {
            dicsStatic = new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            dicsDynamic = new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        function staticSceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.static')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 2] = 'counter';
                        break;
                    case 1:
                        parts[parts.length - 2] = 'bicycle';
                        break;
                    case 2:
                        parts[parts.length - 2] = 'bonsai';
                        break;
                    case 3:
                        parts[parts.length - 2] = 'kitchen';
                        break;
                    case 4:
                        parts[parts.length - 2] = 'smoke';
                        break;
                    case 5:
                        parts[parts.length - 2] = 'explosion';
                        break;
                    case 6:
                        parts[parts.length - 2] = 'bunny';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("static-reconstruction").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsStatic.medias = dicsStatic._getMedias();
        }


        function dynamicSceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.dynamic')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 2] = 'cloud';
                        break;
                    case 1:
                        parts[parts.length - 2] = 'heart';
                        break;
                    case 2:
                        parts[parts.length - 2] = 'suzanne';
                        break;
                    case 3:
                        parts[parts.length - 2] = 'trex';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("dynamic-reconstruction").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsDynamic.medias = dicsDynamic._getMedias();
        }

