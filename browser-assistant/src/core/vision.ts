// When a question needs eyes. A model that can see gets a picture of the page with such a question even if the
// 📷 button is off — "what is in this image?" should not need a second try. Words in English, Russian and Spanish.

const LOOKING = new RegExp(
  "(?<![\\p{L}])(image|images|picture|pictures|photo|photos|screenshot|screen|look|looks|see|visible|chart|diagram|" +
  "logo|icon|colou?r|изображ\\p{L}*|картин\\p{L}*|фото\\p{L}*|снимк?\\p{L}*|скриншот\\p{L}*|экран\\p{L}*|рисун\\p{L}*|" +
  "видишь|видно|выгляд\\p{L}*|график\\p{L}*|схем\\p{L}*|логотип\\p{L}*|цвет\\p{L}*|imagen\\p{L}*|foto\\p{L}*|" +
  "captura|pantalla|ves|dibujo)(?![\\p{L}])", "iu");

export function wantsToSee(question: string | undefined): boolean {
  return !!question && LOOKING.test(question);
}
