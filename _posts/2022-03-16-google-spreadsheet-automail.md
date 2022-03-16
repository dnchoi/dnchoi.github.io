---
layout: post
title: Google Spread sheet & Auto send to mail
author: dnchoi
date: 2022-03-13 09:45:47
categories: [etc]
tags: [etc]
---

# Google Spread sheet & Auto send to mail

## 1. Google API registration
## 2. Name the scope in google spread sheet
### 전송하고자 하는 범위를 지정해야한다.
1. 상단의 데이터 탭
2. 이름이 지정된 범위
3. 범위 추가

![Untitled_1](assets/img/post/google-spreadsheet/Untitled_1.png)

|:---:|
|Stock 제목|
|사용할 sheet!범위A:범위B|

## 3. Create Apps Script
> myFunction.gs
```javascript
function myFunction() {
  sendEmail()
}
 
function sendEmail() {
  var stockData = getData("table", "Stocks", true);
  var stockData2 = getData("2022", "send_day", false);
  // console.warn(stockData2)
  var stockColor = getColor();
  var body = getEmailText(stockData2,stockColor);
  // console.warn(body)
  var htmlBody = getEmailHtml(stockData,stockColor);
  // console.warn(htmlBody)
  let today = new Date();   

  var week = new Array("일", "월", "화", "수", "목", "금", "토")
  let year = today.getFullYear(); // 년도
  let month = today.getMonth() + 1;  // 월
  let date = today.getDate();  // 날짜
  let day = today.getDay();  // 요일

  let key = false
  var que = null
  stockData2.forEach(function(value) {
    let a = value["공휴일"]
    let y = value["날짜"].getFullYear(); // 년도
    let m = value["날짜"].getMonth() + 1;  // 월
    let da = value["날짜"].getDate();  // 날짜
    let d = value["날짜"].getDay();  // 요일

    que = y+"-"+m+"-"+da+"-"+week[d]
    var now = year+"-"+month+"-"+date+"-"+week[day]
    // console.log(que, now)
    if(que == now){
      if(a == 1){
        key = true
      }
      else{
        key = false
      }
    }
    else{
      key = false
    }
  })

  var subject_val = "자동 메일 전송 " + year + '년 ' + month + '월 ' + date + "일 (" + week[day] + ")"
  
  var address = "To@email.com"
  if(week[day] != "토" && week[day] != "일" && key == false){
    MailApp.sendEmail({
      to: address,
      subject: subject_val,
      htmlBody : htmlBody
    });
    console.log("Send mail : ", que, week[day], key)
  }
  else{
    console.log("Not send mail : ", que, week[day], key)
  }
}
 
function getEmailText(stockData) {
  var text = "";
  stockData.forEach(function(stock) {
    text = text + stock.날짜 + "\n" + stock.요일 + "\n" + stock.공휴일 + "\n" + "\n-----------------------\n\n";
  });
  return text;
}
 
/**
 * @OnlyCurrentDoc
 */
function getData(t, s, f) {
  var values = SpreadsheetApp.getActive().getSheetByName(t).getRange(s).getValues();
  values.shift(); //remove headers
  var stocks = [];
  if(f==true){
    values.forEach(function(value) {
      var stock = {};
      stock.TitleA = value[0];
      stock.TitleB = value[1];
      stock.TitleC = value[2];
      stock.TitleD = value[3];
      stocks.push(stock);
    })
  }
  else{
    values.forEach(function(value) {
      var stock = {};
      stock.날짜 = value[0];
      stock.요일 = value[1];
      stock.공휴일 = value[2];
      stocks.push(stock);
    })
  }
  return stocks;
}
 
function getColor() {
  var values = SpreadsheetApp.getActive().getSheetByName("table").getRange("Stocks").getBackgrounds();
  values.shift(); //remove headers
  var colors = [];
  values.forEach(function(value) {
    var color = {};
	color.TitleA = value[0];
	color.TitleB = value[1];
	color.TitleC = value[2];
	color.TitleD = value[3];
    colors.push(color);
  })
  return colors;
}
 
 
 
function getEmailHtml(stockData,stockColor) {
  var htmlTemplate = HtmlService.createTemplateFromFile("Template.html");
  htmlTemplate.stocks = stockData; 
  htmlTemplate.colors = stockColor; 
  var htmlBody = htmlTemplate.evaluate().getContent();
  return htmlBody;
}

```

> Template.html
```html
<!DOCTYPE html>
<html>
  <head>
    <base target="_top">
  </head>
  <body>
    <span
      style="
      font-size: 1.5em;
      line-height: 1.0em;  
      color: black;
      font-family: arial;
    ">
      안녕하세요.<br><br>
      자동 메일 전송 테스트 입니다.<br><br>  
    </span>
    <div dir="ltr">
      <table cellspacing="0" cellpadding="0" dir="ltr" border="1" style="table-layout:fixed;font-size:8pt;font-family:Arial;border-collapse:collapse;border:none">
          <colgroup>
              <col width="100">
                  <col width="100">
                      <col width="700">
                          <col width="100">
          </colgroup>
          <tbody>
              <tr style="height:21px">
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;background-color:rgb(207,226,243);font-size:11pt;color:rgb(0,0,0);text-align:center"
                  rowspan="1" colspan="12">자동 메일 전송 프로그램 테스트</td>
              <tr style="height:21px">
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;background-color:rgb(207,226,243);font-size:12pt;text-align:center">TitleA</td>
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;background-color:rgb(207,226,243);font-size:12pt;text-align:center">TitleB</td>
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;background-color:rgb(207,226,243);font-size:12pt;text-align:center">TitleC</td>
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;background-color:rgb(207,226,243);font-size:12pt;text-align:center">TitleD</td>
              </tr>
              <? for(var i = 0; i < stocks.length; i++) { ?>
              <tr style="height:100%">
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;font-size:10pt;text-align:center; background-color : <?= colors[i].TitleA?>"><pre><?= stocks[i].TitleA?></pre></td>
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;font-size:10pt;text-align:center; background-color : <?= colors[i].TitleB?>"><pre><?= stocks[i].TitleB?></pre></td>
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;font-size:10pt;text-align:bottom; background-color : <?= colors[i].TitleC?>"><pre><?= stocks[i].TitleC?></pre></td>
                  <td style="border-width:1px;border-style:solid;overflow:hidden;vertical-align:center;font-size:10pt;text-align:bottom; background-color : <?= colors[i].TitleD?>"><pre><?= stocks[i].TitleD?></pre></td>
              </tr>
              <? } ?> 
          </tbody>
      </table>
  </div>
  <span
    style="
    font-size: 1.5em;
    line-height: 1.0em;  
    color: black;
    font-family: arial;
  ">
    <br><br>
    감사합니다.
    <br><br>
  </span>
  </body>
</html>
```
