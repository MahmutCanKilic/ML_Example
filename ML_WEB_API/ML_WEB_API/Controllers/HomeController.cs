using Business;
using Microsoft.AspNetCore.Mvc;
using Org.BouncyCastle.Crypto.Agreement;

namespace ML_WEB_API.Controllers
{
    public class HomeController : Controller
    {
        private readonly DataSetBusiness business;

        public HomeController(DataSetBusiness business)
        {
            this.business = business;
        }

        [HttpGet(nameof(GetDatas))]
        public IActionResult GetDatas()
        {
            var data = business.GetDatas();
            return Ok(data.Data);
        }
    }
}
